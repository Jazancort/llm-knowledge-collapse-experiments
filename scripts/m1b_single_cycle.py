"""M1B: Single Training Cycle.

Protocol:
1. Generate synthetic answers from base model (Training Seed, NOT Probe Set)
2. QLoRA fine-tune (1 epoch, q_proj + v_proj, r=16)
3. Load adapter dynamically (NO merge into 4-bit base)
4. Re-evaluate: accuracy, confidence, entropy, CKA-Global, CKA-Factual
5. Measure effective rank of LoRA matrices

Key question: Does CKA-Factual move after a single fine-tuning cycle?

Run: uv run python scripts/m1b_single_cycle.py
"""

import sys
import time
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
from datasets import Dataset
import re

# --- Config ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MONITOR_LAYERS = [1, 7, 13, 19, 25]
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "m1b"

# Training seed (SEPARATE from probe set)
TRAINING_SEED = [
    "What is the capital of Germany?",
    "Who discovered penicillin?",
    "What is the speed of sound?",
    "Who painted Starry Night?",
    "What is the atomic number of carbon?",
    "Who wrote Pride and Prejudice?",
    "What is the largest continent?",
    "What year was the internet invented?",
    "Who composed the Four Seasons?",
    "What is the chemical formula for water?",
    "What planet has the most moons?",
    "Who invented the light bulb?",
    "What is the tallest mountain?",
    "What year did the Titanic sink?",
    "Who discovered gravity?",
    "What is the smallest country?",
    "What language has the most speakers?",
    "Who wrote The Odyssey?",
    "What is the largest desert?",
    "What year was the first airplane flight?",
]

# Probe set (QUARANTINED - evaluation only)
PROBE_SET = [
    {"question": "What is the capital of France?", "answers": ["Paris"]},
    {"question": "Who wrote Romeo and Juliet?", "answers": ["William Shakespeare", "Shakespeare"]},
    {"question": "What is the chemical symbol for gold?", "answers": ["Au"]},
    {"question": "What planet is closest to the Sun?", "answers": ["Mercury"]},
    {"question": "Who painted the Mona Lisa?", "answers": ["Leonardo da Vinci", "Da Vinci", "Leonardo"]},
    {"question": "What is the largest ocean on Earth?", "answers": ["Pacific Ocean", "Pacific"]},
    {"question": "What year did World War II end?", "answers": ["1945"]},
    {"question": "Who developed the theory of relativity?", "answers": ["Albert Einstein", "Einstein"]},
    {"question": "What is the smallest prime number?", "answers": ["2", "two"]},
    {"question": "What is the capital of Japan?", "answers": ["Tokyo"]},
]

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
                self.buffer[idx]["global"].append(t[i, :end+1, :].mean(0).cpu().float().numpy())
                self.buffer[idx]["f1"].append(t[i, end, :].cpu().float().numpy())
        return fn

    def register(self):
        # Navigate to decoder layers regardless of PEFT wrapper
        m = self.model
        if hasattr(m, "base_model"):  # PeftModel wraps with base_model
            m = m.base_model
        # Try common paths
        if hasattr(m, "model") and hasattr(getattr(m, "model"), "layers"):
            layers_container = m.model.layers
        elif hasattr(m, "layers"):
            layers_container = m.layers
        else:
            # Walk children to find ModuleList named 'layers'
            for name, child in m.named_modules():
                if name.endswith("layers") and hasattr(child, "__getitem__"):
                    layers_container = child
                    break
            else:
                raise RuntimeError(f"Cannot find decoder layers in {type(m)}")

        for idx in self.layers:
            self.hooks.append(layers_container[idx].register_forward_hook(self._hook(idx)))

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def reset(self):
        self.buffer = {l: {"global": [], "f1": []} for l in self.layers}

    def stacked(self, idx, key):
        return np.stack(self.buffer[idx][key])


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtX = X @ X.T
    YtY = Y @ Y.T
    num = np.trace(XtX @ YtY)
    den = np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro')
    return float(num / (den + 1e-10))


def evaluate(model, tokenizer, extractor, probe_set):
    """Run probe set: extract representations via forward, then generate for accuracy."""
    extractor.reset()
    predictions, log_probs, entropies = [], [], []

    for item in probe_set:
        msgs = [
            {"role": "system", "content": "Answer the following question in 5 words or less."},
            {"role": "user", "content": item["question"]},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        end_idx = inputs.input_ids.shape[1] - 1
        extractor.end_indices = [end_idx]

        # Forward for representations (hooks active)
        with torch.no_grad():
            model(**inputs)

        # Disable hooks for generate (avoids index errors on single-token steps)
        extractor.clear()

        # Generate for accuracy (DETERMINISTIC - no sampling noise)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
        gen_ids = out.sequences[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        predictions.append(text)

        lps, ents = [], []
        for i, logits in enumerate(out.scores):
            if i >= len(gen_ids):
                break
            lf = logits[0].float()
            lps.append(torch.log_softmax(lf, -1)[gen_ids[i].item()].item())
            ents.append(Categorical(logits=lf).entropy().item())
        if lps:
            log_probs.append(np.mean(lps))
            entropies.append(np.mean(ents))

        # Re-register hooks for next sample
        extractor.register()

    correct = sum(exact_match(p, item["answers"]) for p, item in zip(predictions, probe_set))
    return {
        "accuracy": correct / len(probe_set),
        "avg_log_prob": np.mean(log_probs) if log_probs else 0,
        "avg_entropy": np.mean(entropies) if entropies else 0,
    }


def compute_lora_spectrum(model):
    """Compute effective rank and norms of LoRA matrices."""
    ranks = []
    spectral_norms = []
    frobenius_norms = []

    for name, param in model.named_parameters():
        if "lora_A" in name and param.requires_grad:
            # Find corresponding B matrix
            b_name = name.replace("lora_A", "lora_B")
            b_param = dict(model.named_parameters()).get(b_name)
            if b_param is not None:
                A = param.data.float().cpu()
                B = b_param.data.float().cpu()
                AB = B @ A  # (out_dim, in_dim) approximately
                svs = torch.linalg.svdvals(AB)
                svs_norm = svs / (svs.sum() + 1e-10)
                eff_rank = torch.exp(-(svs_norm * torch.log(svs_norm + 1e-10)).sum()).item()
                ranks.append(eff_rank)
                spectral_norms.append(svs[0].item())
                frobenius_norms.append(torch.norm(AB, 'fro').item())

    return {
        "effective_rank_mean": np.mean(ranks) if ranks else 0,
        "spectral_norm_mean": np.mean(spectral_norms) if spectral_norms else 0,
        "frobenius_norm_mean": np.mean(frobenius_norms) if frobenius_norms else 0,
        "n_lora_pairs": len(ranks),
    }


def main():
    print("=" * 60)
    print("M1B: SINGLE TRAINING CYCLE")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load base model ---
    print("\n[1/6] Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    model.eval()

    # --- Gen0 evaluation ---
    print("\n[2/6] Gen0 evaluation (baseline)...")
    extractor = ProbeExtractor(model, MONITOR_LAYERS)
    extractor.register()
    gen0_metrics = evaluate(model, tokenizer, extractor, PROBE_SET)
    gen0_reps = {idx: {"global": extractor.stacked(idx, "global"),
                       "f1": extractor.stacked(idx, "f1")} for idx in MONITOR_LAYERS}
    extractor.clear()
    print(f"  Accuracy: {gen0_metrics['accuracy']:.2%}")
    print(f"  Log-prob: {gen0_metrics['avg_log_prob']:.4f}")
    print(f"  Entropy:  {gen0_metrics['avg_entropy']:.4f}")

    # --- Generate synthetic data ---
    print("\n[3/6] Generating synthetic training data...")
    synthetic_texts = []
    for q in TRAINING_SEED:
        msgs = [
            {"role": "system", "content": "Answer the following question in 5 words or less."},
            {"role": "user", "content": q},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, temperature=0.7, do_sample=True, top_p=0.9)
        full_text = tokenizer.decode(out[0], skip_special_tokens=False)
        synthetic_texts.append(full_text)

    print(f"  Generated {len(synthetic_texts)} training samples")

    # --- QLoRA setup ---
    print("\n[4/6] Setting up QLoRA (r=16, q_proj + v_proj)...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Prepare dataset ---
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

    ds = Dataset.from_dict({"text": synthetic_texts})
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds.set_format("torch")

    # --- Train ---
    print("\n[5/6] Training (1 epoch)...")
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=peft_model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # --- Gen1 evaluation ---
    print("\n[6/6] Gen1 evaluation (after fine-tuning)...")
    peft_model.eval()
    extractor = ProbeExtractor(peft_model, MONITOR_LAYERS)
    extractor.register()
    gen1_metrics = evaluate(peft_model, tokenizer, extractor, PROBE_SET)
    gen1_reps = {idx: {"global": extractor.stacked(idx, "global"),
                       "f1": extractor.stacked(idx, "f1")} for idx in MONITOR_LAYERS}
    extractor.clear()

    # LoRA spectrum
    lora_spectrum = compute_lora_spectrum(peft_model)

    # --- CKA comparison ---
    print("\n" + "=" * 60)
    print("M1B RESULTS: Gen0 vs Gen1")
    print("=" * 60)

    print(f"\n  {'Metric':<20} {'Gen0':<12} {'Gen1':<12} {'Delta'}")
    print(f"  {'-'*56}")
    print(f"  {'Accuracy':<20} {gen0_metrics['accuracy']:<12.2%} {gen1_metrics['accuracy']:<12.2%} {gen1_metrics['accuracy']-gen0_metrics['accuracy']:+.2%}")
    print(f"  {'Log-prob':<20} {gen0_metrics['avg_log_prob']:<12.4f} {gen1_metrics['avg_log_prob']:<12.4f} {gen1_metrics['avg_log_prob']-gen0_metrics['avg_log_prob']:+.4f}")
    print(f"  {'Entropy':<20} {gen0_metrics['avg_entropy']:<12.4f} {gen1_metrics['avg_entropy']:<12.4f} {gen1_metrics['avg_entropy']-gen0_metrics['avg_entropy']:+.4f}")

    print(f"\n  CKA (Gen0 vs Gen1):")
    print(f"  {'Layer':<8} {'CKA-Global':<14} {'CKA-Factual':<14} {'Factual<Global?'}")
    print(f"  {'-'*50}")
    for idx in MONITOR_LAYERS:
        cka_g = linear_cka(gen0_reps[idx]["global"], gen1_reps[idx]["global"])
        cka_f = linear_cka(gen0_reps[idx]["f1"], gen1_reps[idx]["f1"])
        more_sensitive = "YES" if cka_f < cka_g else "no"
        print(f"  {idx:<8} {cka_g:<14.6f} {cka_f:<14.6f} {more_sensitive}")

    print(f"\n  LoRA Adapter Health:")
    print(f"    Effective Rank (mean): {lora_spectrum['effective_rank_mean']:.2f}")
    print(f"    Spectral Norm (mean):  {lora_spectrum['spectral_norm_mean']:.6f}")
    print(f"    Frobenius Norm (mean): {lora_spectrum['frobenius_norm_mean']:.6f}")
    print(f"    LoRA pairs measured:   {lora_spectrum['n_lora_pairs']}")

    # Key question
    cka_f_mid = linear_cka(gen0_reps[13]["f1"], gen1_reps[13]["f1"])
    cka_g_mid = linear_cka(gen0_reps[13]["global"], gen1_reps[13]["global"])
    moved = cka_f_mid < 0.999
    print(f"\n  KEY QUESTION: Does CKA-Factual move after 1 cycle?")
    print(f"    CKA-Factual (layer 13): {cka_f_mid:.6f}")
    print(f"    CKA-Global  (layer 13): {cka_g_mid:.6f}")
    print(f"    Answer: {'YES - signal detected' if moved else 'NO - insufficient resolution'}")

    print(f"\n  M1B: {'PASSED' if moved else 'NEEDS REVIEW'}")


if __name__ == "__main__":
    main()
