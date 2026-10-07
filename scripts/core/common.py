"""Shared protocol constants and primitives for the Intervention Window paper.

Every constant here is inherited verbatim from the ETP scripts so that the
trajectories are calibration-comparable (same shape, same slope), even though
the K0 set is re-frozen and therefore not item-comparable. Sources:

  qwen   : scripts/causal_intervention.py, scripts/c5_replicate_masks.py
  gemma  : scripts/exp_gemma3_extra_seeds.py

Deliberate departures from ETP, all documented in docs/EXPERIMENT_PLAN.md:
  EVAL_SIZE 200 -> 400   (§5.1, cuts the MDE from ~5pp to ~3.5pp)
  GENERATIONS 5 -> 15    (§2, the recovery horizon this paper needs)
  absolute -> relative dose for the C5 intervention (§5.4)
  batched K0 eval, larger generation batch (throughput; see below)

On batch sizes and reproducibility:
  - K0 evaluation is GREEDY, so batching is verdict-identical to the ETP
    item-at-a-time loop (left padding + explicit pad_token_id). Pure speedup.
  - Synthetic generation is SAMPLED, so the batch size changes which random
    draws land on which prompt. Corpora are therefore not bit-identical to the
    ETP ones. Acceptable here because this paper re-freezes K0 and regenerates
    its own baseline (docs/DEPENDENCIES.md) — but the batch size is a CONSTANT
    across every condition, so within-paper comparisons stay clean. Never
    change gen_batch mid-study.

The ETP scripts themselves are NOT imported or edited: they are the
reproducibility artefact of a submitted paper.
"""
import gc, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # for io_utils

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

# Crash-safe I/O lives in a torch-free module so test_logic.py can exercise the
# real functions instead of a copy. Re-exported: callers import it from here.
from io_utils import (clean_stale_tmp, load_json, resume_point,  # noqa: F401
                      save_json)

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------- protocol
DATASET = ("mandarjoshi/trivia_qa", "rc.nocontext")
DATASET_SEED = 15          # ds.shuffle(seed=15) in every ETP script
TRAIN_SIZE = 2000
EVAL_SIZE = 400            # ETP used 200 — see §5.1
GENERATIONS = 15           # ETP used 5/10 — see §2
LR = 1e-5
EPOCHS = 2
MAX_LENGTH = 256

BACKBONES = {
    "qwen": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "rank": 256,
        "seed_base": 15,
        "chat_style": "system",   # separate system message
        "gen_batch": 32,          # ETP used 16; raised for the RTX 5000 (32 GB)
        "eval_batch": 32,
        # Training geometry is FIXED, never memory-adaptive: see get_batch_size.
        # causal_intervention.py:190, the branch a >=20 GB card takes.
        "train_batch": 4, "train_accum": 4,
        # MEASURED, not guessed: training at 4x4 peaks at 6.45 GB reserved
        # (2026-07-27, seed 137 Gen1). The old 12.0 came from an OOM that happened
        # under DataParallel (§5.7) at DOUBLE this batch, and it was costing real
        # runs: with LM Studio holding ~8 GB of GPU 0, three trajectories aborted
        # at "only 11.3 GB free" on a card where 6.45 GB would have done.
        # 9.0 = measured peak + ~40% for fragmentation and the eval/generate
        # phases. Refusing to run is the right behaviour (never shrink the batch,
        # §5.6) but the threshold has to be true or it refuses valid work.
        "min_free_gb": 9.0,       # r=256 at 4x4, MAX_LENGTH=256; pico medido 6,45

    },
    "llama": {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "rank": 16,
        "seed_base": 15,
        "chat_style": "system",   # Llama-3.2 has system role
        "gen_batch": 16,
        "eval_batch": 32,
        # Same batch geometry as Gemma: 8x2 = effective batch 16.
        # Llama-3.2-1B is comparable to gemma-3-1b in VRAM at r=16.
        "train_batch": 8, "train_accum": 2,
        "min_free_gb": 10.0,      # r=16, measured conservative estimate
    },
    "gemma": {
        "model": "google/gemma-3-1b-it",
        "rank": 16,
        "seed_base": 15,
        "chat_style": "user",     # gemma-3-1b-it has no system role
        "gen_batch": 16,          # ETP used 8
        "eval_batch": 32,
        "train_batch": 8, "train_accum": 2,   # exp_gemma3_extra_seeds.py:163
        "min_free_gb": 10.0,      # r=16 is far lighter than Qwen's r=256
    },
    "sheared": {
        # Sheared-LLaMA-1.3B-ShareGPT: pruned from LLaMA-2-7B (Xia et al., ICLR 2024)
        # No KD objective -- fills the (pruning, no-KD) cell of the provenance 2x2.
        # Instruction recipe: 10k ShareGPT pairs, single-turn only (declared confound).
        # Tokenizer: LLaMA-2 (32k vocab); no chat_template -> Alpaca format.
        "model": "princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT",
        "rank": 16,
        "seed_base": 15,
        "chat_style": "alpaca",   # no chat_template; uses ### Instruction / Input / Response
        "gen_batch": 16,
        "eval_batch": 32,
        "train_batch": 8, "train_accum": 2,
        "min_free_gb": 10.0,
    },
}

SYSTEM_PROMPT = "Answer the following question in 5 words or less."

NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                "ten": "10"}


# ---------------------------------------------------------------- scoring
def normalize_answer(text):
    text = text.lower().strip()
    for w, d in NUMBER_WORDS.items():
        text = re.sub(rf"\b{w}\b", d, text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred, gts):
    p = normalize_answer(pred)
    return any(normalize_answer(g) in p or p in normalize_answer(g) for g in gts)


# Native ShareGPT/Alpaca template for Sheared-LLaMA-1.3B-ShareGPT.
# Source: model card at princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT.
# Input field receives SYSTEM_PROMPT + "\n" + question, mirroring the
# system-role prepend used by other backbones.
_SHAREPOINT_TEMPLATE = (
    "You are a helpful assistant. "
    "Write a response that appropriately completes the request.\n\n"
    "### Input:\n{input}\n\n### Response:\n"
)


def load_tokenizer(cfg):
    """Single tokenizer loader used by EVERY script (freeze_k0, factorial, G2).
    Sheared-LLaMA requires LlamaTokenizer (slow path) to avoid the
    AutoTokenizer -> tiktoken conversion failure on its SentencePiece file.
    All others use AutoTokenizer.
    Assertion: token IDs for three fixed prompts must match between slow and
    fast tokenizers (or between any two calls) — prevents silent divergence.
    """
    from transformers import AutoTokenizer, LlamaTokenizer
    model = cfg["model"]
    if cfg.get("chat_style") == "alpaca":
        tok = LlamaTokenizer.from_pretrained(model, use_fast=False)
    else:
        tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def format_prompt(tokenizer, q, cfg):
    if cfg["chat_style"] == "alpaca":
        # Native ShareGPT format: system prompt prepended to question in Input.
        return _SHAREPOINT_TEMPLATE.format(input=f"{SYSTEM_PROMPT}\n{q}")
    if cfg["chat_style"] == "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}]
    else:
        msgs = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n{q}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------- data
def load_trivia(eval_size=EVAL_SIZE):
    ds = load_dataset(*DATASET, split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=DATASET_SEED)
    train_q = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_q = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + eval_size)]
    eval_a = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]]
              for i in range(TRAIN_SIZE, TRAIN_SIZE + eval_size)]
    return train_q, eval_q, eval_a


def load_k0(backbone, eval_q, eval_a):
    """Load the frozen canonical K0. Fails loudly — never silently falls back."""
    path = ROOT / "outputs" / f"k0_canonical_{backbone}.json"
    if not path.exists():
        raise SystemExit(
            f"Canonical K0 not frozen for '{backbone}'.\n"
            f"Run first:  uv run python scripts/freeze_k0.py --backbone {backbone}"
        )
    meta = load_json(path, expect=dict)
    if meta["eval_size"] > len(eval_q):
        raise SystemExit(f"K0 was frozen at EVAL_SIZE={meta['eval_size']}, "
                         f"but only {len(eval_q)} eval questions were loaded.")
    idx = meta["k0_indices"]
    questions = [eval_q[i] for i in idx]
    # The frozen file stores the questions; verify the dataset ordering matches.
    if "questions_sha" in meta:
        import hashlib
        got = hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest()[:16]
        if got != meta["questions_sha"]:
            raise SystemExit(
                "K0 question mismatch: the dataset ordering changed since the freeze.\n"
                f"  expected sha {meta['questions_sha']}, got {got}\n"
                "Do not mix bases — re-freeze, or pin the datasets version."
            )
    return questions, [eval_a[i] for i in idx], meta


# ---------------------------------------------------------------- model
def load_base(cfg, quantized=True, device=None):
    # Pin to one GPU rather than device_map="auto": Athena is a 2-GPU box shared
    # with LM Studio, and "auto" would happily shard a 1.5B model across both,
    # making every forward pass pay a PCIe hop.
    dev = torch.cuda.current_device() if device is None else device
    kwargs = dict(device_map={"": dev}, torch_dtype=torch.bfloat16)
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **kwargs)
    tok = load_tokenizer(cfg)  # single loader; handles alpaca/LlamaTokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()
    return model, tok


def evaluate_k0(model, tokenizer, questions, answers, cfg, batch_size=None):
    """Greedy, max_new_tokens=20, inference_mode, explicit pad_token_id (§5.1).

    Batched. Greedy decoding with left padding is invariant to batch size, so
    this is a pure speedup: identical verdicts, ~5-8x less wall clock than the
    item-at-a-time loop in the ETP scripts. With EVAL_SIZE=400 that loop would
    have dominated the per-generation cost.

    Set batch_size=1 to reproduce the ETP call pattern exactly.
    """
    bs = cfg.get("eval_batch", 32) if batch_size is None else batch_size
    assert tokenizer.padding_side == "left", \
        "K0 eval requires left padding, else generated text is offset"
    results = []
    for i in tqdm(range(0, len(questions), bs), desc="    K0", leave=False):
        qb, ab = questions[i:i + bs], answers[i:i + bs]
        prompts = [format_prompt(tokenizer, q, cfg) for q in qb]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=20, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for text, gts in zip(texts, ab):
            results.append(exact_match(text.strip(), gts))
    return results


def generate_synthetic(model, tokenizer, questions, cfg, seed_offset=0):
    torch.manual_seed(seed_offset)
    bs = cfg["gen_batch"]
    synthetic = []
    for i in tqdm(range(0, len(questions), bs), desc="    synth", leave=False):
        batch = questions[i:i + bs]
        prompts = [format_prompt(tokenizer, q, cfg) for q in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=30, temperature=0.7,
                                 do_sample=True, top_p=0.9,
                                 pad_token_id=tokenizer.pad_token_id)
        for seq in out:
            synthetic.append(tokenizer.decode(seq, skip_special_tokens=False))
    return synthetic


# ---------------------------------------------------------------- metrics
ASSISTANT_PATTERNS = [
    r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)",   # qwen
    r"<start_of_turn>model\n(.*?)(?:<end_of_turn>|$)",      # gemma
]


def extract_responses(texts):
    out = []
    for text in texts:
        for pat in ASSISTANT_PATTERNS:
            m = re.findall(pat, text, re.DOTALL)
            if m:
                out.append(m[-1].strip())
                break
    return out


def mtld(tokens, threshold=0.72):
    """Measure of Textual Lexical Diversity — length-independent, unlike TTR."""
    if len(tokens) < 10:
        return float("nan")

    def one_pass(seq):
        factors, start = 0.0, 0
        for i in range(len(seq)):
            window = seq[start:i + 1]
            if len(set(window)) / len(window) <= threshold:
                factors += 1
                start = i + 1
        rest = seq[start:]
        if rest:
            ttr = len(set(rest)) / len(rest)
            factors += (1 - ttr) / (1 - threshold) if threshold < 1 else 0
        # factors < 1 means the text never dropped to the TTR threshold, i.e.
        # maximum diversity. Clamping to 1 makes MTLD = len(seq) there, which is
        # the documented ceiling; leaving it at 0 would yield nan for exactly the
        # most diverse inputs.
        return len(seq) / max(factors, 1.0)

    return float(np.mean([one_pass(tokens), one_pass(tokens[::-1])]))


def compute_diversity(texts):
    """distinct-1, mean_length, unique_responses (as in ETP) + MTLD."""
    responses = extract_responses(texts)
    if not responses:
        return {}
    all_tokens = [t for r in responses for t in r.lower().split()]
    total = len(all_tokens)
    if total == 0:
        return {}
    lengths = [len(r.split()) for r in responses]
    return {
        "distinct_1": round(len(set(all_tokens)) / total, 4),
        "mean_length": round(sum(lengths) / len(lengths), 2),
        "unique_responses": len(set(r.lower().strip() for r in responses)),
        "mtld": round(mtld(all_tokens), 2),
        "n_responses": len(responses),
    }


def compute_lora_spectrum(model):
    """Adapter scalars, identical definition to g1_rank_ablation.py:109-122.

    Present in the ETP g1_* files but absent from every intervention file —
    recorded here for every generation (docs/BASELINE_DATA.md §4).
    """
    params = dict(model.named_parameters())
    ranks, s_norms, f_norms = [], [], []
    for name, p in params.items():
        if "lora_A" in name and p.requires_grad:
            b = params.get(name.replace("lora_A", "lora_B"))
            if b is None:
                continue
            AB = b.data.float().cpu() @ p.data.float().cpu()
            svs = torch.linalg.svdvals(AB)
            svs_n = svs / (svs.sum() + 1e-10)
            ranks.append(torch.exp(-(svs_n * torch.log(svs_n + 1e-10)).sum()).item())
            s_norms.append(svs[0].item())
            f_norms.append(torch.norm(AB, "fro").item())
    if not ranks:
        return {}
    return {"eff_rank": round(float(np.mean(ranks)), 3),
            "spectral": round(float(np.mean(s_norms)), 4),
            "frobenius": round(float(np.mean(f_norms)), 4)}


# ---------------------------------------------------------------- misc
def defrag():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    time.sleep(2)


def get_batch_size(cfg, device=0):
    """Micro-batch / grad-accum. A PER-BACKBONE CONSTANT, not memory-adaptive.

    This used to size off free memory, and that was a design error. Athena's two
    cards have different free memory (GPU 0 ~23 GB, GPU 1 ~14 GB), so the
    baseline on GPU 0 got 8x2 while the W5 arm on GPU 1 got 2x8. The effective
    batch is 16 either way, but micro-batch size is NOT a free parameter here:

      - lora_dropout=0.05 draws from the RNG per forward pass, so a different
        micro-batch layout consumes the stream differently;
      - bf16 accumulation order changes with micro-batch size, so gradients
        differ in the low bits even before dropout.

    Neither effect is large, but both are CONFOUNDED WITH CONDITION as soon as
    the arms land on different cards, which is exactly what the 2-GPU schedule
    does. A 2pp intervention effect cannot be defended if the comparison arm was
    also trained with a different batch geometry.

    So it is pinned, per backbone, to what the ETP scripts used — see
    train_batch/train_accum in BACKBONES:
      qwen  : causal_intervention.py:188-192 -> 4, 4  (its >=20 GB branch)
      gemma : exp_gemma3_extra_seeds.py:163  -> 8, 2

    Qwen's 4x4 fits in the ~14 GB free on the smaller card (2x8 measured ~6.2 GB
    for r=256 at MAX_LENGTH=256; 4x4 roughly doubles activations). `device` is
    used only for the OOM diagnostic below.
    """
    bs, accum = cfg["train_batch"], cfg["train_accum"]
    need = cfg["min_free_gb"]
    free_gb = torch.cuda.mem_get_info(device)[0] / 1024 ** 3
    if free_gb < need:
        # Refuse rather than silently shrink: shrinking is what created the
        # confound. Freeing memory (or waiting) keeps the study comparable.
        raise SystemExit(
            f"GPU {device} has only {free_gb:.1f} GB free; training at "
            f"{bs}x{accum} needs ~{need:.0f} GB.\n"
            f"  The batch geometry is fixed study-wide and must NOT be reduced "
            f"for one condition — that would confound batch size with condition.\n"
            f"  Free memory in LM Studio, or run on the other card (--gpu N).")
    return bs, accum


def pick_gpu(prefer=None, min_free_gb=10.0):
    """Choose a CUDA device by free memory and pin the process to it.

    Athena has two Ada cards (RTX 5000 32 GB / RTX 4000 20 GB) that are shared
    with LM Studio, so the free split changes between runs. Returns the index
    actually selected, after setting it as the current device.
    """
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible. On Athena, check `nvidia-smi`.")
    n = torch.cuda.device_count()
    stats = []
    for i in range(n):
        free_b, total_b = torch.cuda.mem_get_info(i)
        stats.append((i, free_b / 1024 ** 3, total_b / 1024 ** 3,
                      torch.cuda.get_device_name(i)))
    for i, free, total, nm in stats:
        print(f"  GPU {i}: {nm}  {free:.1f} / {total:.1f} GB free")

    if prefer is not None:
        cand = [s for s in stats if s[0] == prefer]
        if not cand:
            raise SystemExit(f"--gpu {prefer} does not exist (found {n} device(s)).")
        idx, free = cand[0][0], cand[0][1]
        if free < min_free_gb:
            print(f"  WARNING: GPU {idx} has only {free:.1f} GB free "
                  f"(wanted >={min_free_gb:.0f}); proceeding as requested.")
    else:
        idx, free = max(((s[0], s[1]) for s in stats), key=lambda t: t[1])
        if free < min_free_gb:
            raise SystemExit(
                f"No GPU has {min_free_gb:.0f} GB free (best: GPU {idx}, "
                f"{free:.1f} GB). Free memory in LM Studio or pass --gpu N to force.")
        print(f"  -> selected GPU {idx} ({free:.1f} GB free)")

    torch.cuda.set_device(idx)
    return idx
