"""M2: 3-Generation Recursive Training Pilot.

Protocol (D56 - data-only recursion):
- Gen 0: Base model generates synthetic answers for Training Seed
- Gen N: Train NEW adapter from scratch on synthetic data from Gen N-1
- Gen N: Generate new synthetic data using Base + adapter_N
- Evaluate all metrics at each generation
- Clean VRAM between generations

Dataset: TriviaQA (10k train / 1k eval / 200 probe)
Evaluation: Deterministic (do_sample=False)
Adapter: QLoRA r=16, q_proj + v_proj, trained from scratch each gen

Run: uv run python scripts/m2_pilot.py
"""

import sys
import gc
import json
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.distributions import Categorical
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, Dataset

# --- Config ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MONITOR_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 26]
NUM_GENERATIONS = 3
TRAIN_SIZE = 2000  # Start smaller for pilot (scale to 10k in M3)
EVAL_SIZE = 200
PROBE_SIZE = 100
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "m2_pilot"
SEED = 15

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


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


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtX, YtY = X @ X.T, Y @ Y.T
    return float(np.trace(XtX @ YtY) / (np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro') + 1e-10))


class ProbeExtractor:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.hooks = []
        self.buffer = {l: {"global": [], "f1": []} for l in layers}
        self.end_indices = []

    def _hook(self, idx):
        def fn(module, inp, out):
            t = out[0].detach()
            for i in range(t.size(0)):
                end = self.end_indices[i]
                if end >= t.size(1):
                    end = t.size(1) - 1
                self.buffer[idx]["global"].append(t[i, :end+1, :].mean(0).cpu().float().numpy())
                self.buffer[idx]["f1"].append(t[i, end, :].cpu().float().numpy())
        return fn

    def register(self):
        m = self.model
        if hasattr(m, "base_model"):
            m = m.base_model
        for name, child in m.named_modules():
            if name.endswith("layers") and hasattr(child, "__getitem__"):
                for idx in self.layers:
                    self.hooks.append(child[idx].register_forward_hook(self._hook(idx)))
                return
        raise RuntimeError("Cannot find decoder layers")

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def reset(self):
        self.buffer = {l: {"global": [], "f1": []} for l in self.layers}

    def stacked(self, idx, key):
        return np.stack(self.buffer[idx][key]) if self.buffer[idx][key] else np.zeros((1, 1))


def load_base_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_prompt(tokenizer, question):
    msgs = [
        {"role": "system", "content": "Answer the following question in 5 words or less."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_synthetic(model, tokenizer, questions, batch_size=8):
    """Generate synthetic answers for a list of questions. Returns full formatted texts."""
    synthetic = []
    model.eval()
    total = len(questions)
    for i in range(0, total, batch_size):
        batch_q = questions[i:i+batch_size]
        prompts = [format_prompt(tokenizer, q) for q in batch_q]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, temperature=0.7, do_sample=True, top_p=0.9)
        for seq in out:
            synthetic.append(tokenizer.decode(seq, skip_special_tokens=False))
        if (i + batch_size) % 200 == 0 or i + batch_size >= total:
            print(f"    Generated {min(i+batch_size, total)}/{total}...", flush=True)
    return synthetic


def evaluate_accuracy(model, tokenizer, eval_questions, eval_answers):
    """Deterministic evaluation on eval set."""
    model.eval()
    correct = 0
    log_probs_all, entropies_all = [], []
    total = len(eval_questions)

    for idx, (q, answers) in enumerate(zip(eval_questions, eval_answers)):
        prompt = format_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
        gen_ids = out.sequences[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if exact_match(text, answers):
            correct += 1

        lps, ents = [], []
        for j, logits in enumerate(out.scores):
            if j >= len(gen_ids):
                break
            lf = logits[0].float()
            lps.append(torch.log_softmax(lf, -1)[gen_ids[j].item()].item())
            ents.append(Categorical(logits=lf).entropy().item())
        if lps:
            log_probs_all.append(np.mean(lps))
            entropies_all.append(np.mean(ents))

        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"    Eval {idx+1}/{total} (acc so far: {correct}/{idx+1} = {correct/(idx+1):.1%})", flush=True)

    return {
        "accuracy": correct / total,
        "avg_log_prob": np.mean(log_probs_all) if log_probs_all else 0,
        "avg_entropy": np.mean(entropies_all) if entropies_all else 0,
    }


def extract_representations(model, tokenizer, probe_questions, monitor_layers):
    """Extract CKA representations from probe set."""
    extractor = ProbeExtractor(model, monitor_layers)
    extractor.register()

    total = len(probe_questions)
    for idx, q in enumerate(probe_questions):
        prompt = format_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        extractor.end_indices = [inputs.input_ids.shape[1] - 1]
        with torch.no_grad():
            model(**inputs)
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"    Probe {idx+1}/{total}...", flush=True)

    reps = {idx: {"global": extractor.stacked(idx, "global"),
                  "f1": extractor.stacked(idx, "f1")} for idx in monitor_layers}
    extractor.clear()
    return reps


def compute_lora_spectrum(model):
    ranks, s_norms, f_norms = [], [], []
    for name, param in model.named_parameters():
        if "lora_A" in name and param.requires_grad:
            b_name = name.replace("lora_A", "lora_B")
            b_param = dict(model.named_parameters()).get(b_name)
            if b_param is not None:
                AB = (b_param.data.float().cpu() @ param.data.float().cpu())
                svs = torch.linalg.svdvals(AB)
                svs_n = svs / (svs.sum() + 1e-10)
                ranks.append(torch.exp(-(svs_n * torch.log(svs_n + 1e-10)).sum()).item())
                s_norms.append(svs[0].item())
                f_norms.append(torch.norm(AB, 'fro').item())
    return {"eff_rank": np.mean(ranks), "spectral": np.mean(s_norms), "frobenius": np.mean(f_norms)}


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def save_checkpoint(output_dir, gen, results, synthetic_data, reps):
    """Save generation checkpoint to disk."""
    ckpt_dir = output_dir / f"checkpoint_gen{gen}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(ckpt_dir / "synthetic_data.json", "w") as f:
        json.dump(synthetic_data, f)

    # Save reps as npz
    rep_data = {}
    for idx, data in reps.items():
        rep_data[f"layer{idx}_global"] = data["global"]
        rep_data[f"layer{idx}_f1"] = data["f1"]
    np.savez_compressed(ckpt_dir / "representations.npz", **rep_data)

    print(f"  [CHECKPOINT] Saved gen{gen} to {ckpt_dir}", flush=True)


def load_checkpoint(output_dir, gen, monitor_layers):
    """Load checkpoint for a given generation. Returns (results, synthetic_data, reps) or None."""
    ckpt_dir = output_dir / f"checkpoint_gen{gen}"
    if not (ckpt_dir / "results.json").exists():
        return None

    with open(ckpt_dir / "results.json") as f:
        results = json.load(f)

    with open(ckpt_dir / "synthetic_data.json") as f:
        synthetic_data = json.load(f)

    npz = np.load(ckpt_dir / "representations.npz")
    reps = {}
    for idx in monitor_layers:
        reps[idx] = {"global": npz[f"layer{idx}_global"], "f1": npz[f"layer{idx}_f1"]}

    return results, synthetic_data, reps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    print("=" * 60)
    print("M2: 3-GENERATION RECURSIVE TRAINING PILOT")
    print(f"  Seed: {SEED} | Resume: {args.resume}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Load TriviaQA ---
    print("\n[SETUP] Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=SEED)

    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    train_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE)]

    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    probe_questions = [ds[i]["question"] for i in range(TRAIN_SIZE + EVAL_SIZE, TRAIN_SIZE + EVAL_SIZE + PROBE_SIZE)]

    print(f"  Train: {len(train_questions)}, Eval: {len(eval_questions)}, Probe: {len(probe_questions)}")

    # --- Check for existing checkpoints ---
    results = []
    start_gen = 0
    gen0_reps = None
    prev_reps = None
    prev_synthetic = None

    if args.resume:
        # Find latest checkpoint
        for g in range(NUM_GENERATIONS, -1, -1):
            ckpt = load_checkpoint(OUTPUT_DIR, g, MONITOR_LAYERS)
            if ckpt is not None:
                loaded_results, prev_synthetic, prev_reps = ckpt
                results = loaded_results if isinstance(loaded_results, list) else [loaded_results]
                # Load gen0 reps if available
                gen0_ckpt = load_checkpoint(OUTPUT_DIR, 0, MONITOR_LAYERS)
                if gen0_ckpt:
                    _, _, gen0_reps = gen0_ckpt
                start_gen = g + 1
                print(f"\n  [RESUME] Found checkpoint at gen{g}. Resuming from gen{start_gen}.")
                break

    # --- Gen 0: Base model ---
    if start_gen == 0:
        print("\n" + "=" * 60)
        print("GENERATION 0 (Base Model)")
        print("=" * 60)

        model, tokenizer = load_base_model()

        print("  Evaluating accuracy...")
        gen0_metrics = evaluate_accuracy(model, tokenizer, eval_questions, eval_answers)
        print(f"  Accuracy: {gen0_metrics['accuracy']:.2%}")
        print(f"  Log-prob: {gen0_metrics['avg_log_prob']:.4f}")
        print(f"  Entropy:  {gen0_metrics['avg_entropy']:.4f}")

        print("  Extracting representations...")
        gen0_reps = extract_representations(model, tokenizer, probe_questions, MONITOR_LAYERS)

        print("  Generating synthetic data...")
        t0 = time.time()
        synthetic_data = generate_synthetic(model, tokenizer, train_questions)
        print(f"  Generated {len(synthetic_data)} samples in {time.time()-t0:.0f}s")

        gen0_result = {"generation": 0, **gen0_metrics, "adapter": None}
        results = [gen0_result]
        prev_reps = gen0_reps
        prev_synthetic = synthetic_data

        save_checkpoint(OUTPUT_DIR, 0, results, prev_synthetic, gen0_reps)

        del model
        cleanup()
        start_gen = 1

    # --- Generations 1-N ---
    for gen in range(start_gen, NUM_GENERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"GENERATION {gen}")
        print("=" * 60)

        # Reload base
        model, tokenizer = load_base_model()

        # Train new adapter from scratch
        print(f"  Training LoRA on Gen{gen-1} synthetic data...")
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        peft_model = get_peft_model(model, lora_config)

        # Tokenize synthetic data
        def tok_fn(examples):
            return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

        train_ds = Dataset.from_dict({"text": prev_synthetic})
        train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
        train_ds.set_format("torch")

        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / f"gen{gen}"),
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            bf16=True,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            seed=SEED,
        )
        trainer = Trainer(
            model=peft_model, args=args, train_dataset=train_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        t0 = time.time()
        trainer.train()
        train_time = time.time() - t0
        print(f"  Training: {train_time:.0f}s")

        # Evaluate
        print("  Evaluating accuracy...")
        peft_model.eval()
        gen_metrics = evaluate_accuracy(peft_model, tokenizer, eval_questions, eval_answers)
        print(f"  Accuracy: {gen_metrics['accuracy']:.2%}")
        print(f"  Log-prob: {gen_metrics['avg_log_prob']:.4f}")
        print(f"  Entropy:  {gen_metrics['avg_entropy']:.4f}")

        # Representations
        print("  Extracting representations...")
        gen_reps = extract_representations(peft_model, tokenizer, probe_questions, MONITOR_LAYERS)

        # CKA vs Gen0 and vs previous gen
        print("  Computing CKA...")
        cka_vs_gen0 = {}
        cka_vs_prev = {}
        for idx in MONITOR_LAYERS:
            cka_vs_gen0[idx] = {
                "global": linear_cka(gen0_reps[idx]["global"], gen_reps[idx]["global"]),
                "f1": linear_cka(gen0_reps[idx]["f1"], gen_reps[idx]["f1"]),
            }
            cka_vs_prev[idx] = {
                "global": linear_cka(prev_reps[idx]["global"], gen_reps[idx]["global"]),
                "f1": linear_cka(prev_reps[idx]["f1"], gen_reps[idx]["f1"]),
            }

        # Adapter health
        spectrum = compute_lora_spectrum(peft_model)
        print(f"  Effective rank: {spectrum['eff_rank']:.2f}")

        # Generate new synthetic data for next gen
        print("  Generating synthetic data for next gen...")
        t0 = time.time()
        new_synthetic = generate_synthetic(peft_model, tokenizer, train_questions)
        print(f"  Generated {len(new_synthetic)} samples in {time.time()-t0:.0f}s")

        results.append({
            "generation": gen,
            **gen_metrics,
            "adapter": spectrum,
            "cka_vs_gen0": {str(k): v for k, v in cka_vs_gen0.items()},
            "cka_vs_prev": {str(k): v for k, v in cka_vs_prev.items()},
            "train_time": train_time,
        })

        prev_reps = gen_reps
        prev_synthetic = new_synthetic

        # Save checkpoint for this generation
        save_checkpoint(OUTPUT_DIR, gen, results, new_synthetic, gen_reps)

        # Cleanup
        del peft_model, trainer, train_ds
        del model
        cleanup()

    # --- Final Report ---
    print("\n" + "=" * 60)
    print("M2 FINAL REPORT")
    print("=" * 60)

    print(f"\n  {'Gen':<6} {'Accuracy':<12} {'Log-prob':<12} {'Entropy':<12} {'Eff.Rank'}")
    print(f"  {'-'*54}")
    for r in results:
        rank = f"{r['adapter']['eff_rank']:.2f}" if r['adapter'] else "N/A"
        print(f"  {r['generation']:<6} {r['accuracy']:<12.2%} {r['avg_log_prob']:<12.4f} {r['avg_entropy']:<12.4f} {rank}")

    print(f"\n  CKA vs Gen0 (layer 13):")
    print(f"  {'Gen':<6} {'CKA-Global':<14} {'CKA-Factual':<14}")
    print(f"  {'-'*34}")
    for r in results[1:]:
        cka = r["cka_vs_gen0"]["13"]
        print(f"  {r['generation']:<6} {cka['global']:<14.6f} {cka['f1']:<14.6f}")

    # Save results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
