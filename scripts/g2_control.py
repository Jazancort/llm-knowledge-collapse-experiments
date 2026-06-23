"""G2-Control: Single cycle with REAL data to isolate CKA baseline.

Answers the critical question:
  "Does CKA-Factual drop equally when fine-tuning on real vs synthetic data?"

If G2 CKA ≈ G1 CKA (~0.983): drop is generic fine-tuning artifact
If G2 CKA >> G1 CKA (e.g. ~0.995): drop is specific to synthetic recursion

Run: uv run python scripts/g2_control.py
"""

import sys
import gc
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MONITOR_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 26]
SEED = 15
TRAIN_SIZE = 2000
EVAL_SIZE = 200
PROBE_SIZE = 100
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "g2_control"

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
                end = min(self.end_indices[i], t.size(1) - 1)
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

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def reset(self):
        self.buffer = {l: {"global": [], "f1": []} for l in self.layers}

    def stacked(self, idx, key):
        return np.stack(self.buffer[idx][key])


def load_base():
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
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def format_prompt(tokenizer, q):
    msgs = [
        {"role": "system", "content": "Answer the following question in 5 words or less."},
        {"role": "user", "content": q},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_reps(model, tokenizer, questions, layers):
    extractor = ProbeExtractor(model, layers)
    extractor.register()
    for i, q in enumerate(questions):
        prompt = format_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        extractor.end_indices = [inputs.input_ids.shape[1] - 1]
        with torch.no_grad():
            model(**inputs)
        if (i + 1) % 50 == 0:
            print(f"    Probe {i+1}/{len(questions)}...", flush=True)
    reps = {idx: {"global": extractor.stacked(idx, "global"),
                  "f1": extractor.stacked(idx, "f1")} for idx in layers}
    extractor.clear()
    return reps


def evaluate_questions(model, tokenizer, questions, answers):
    correct = 0
    for i, (q, gts) in enumerate(zip(questions, answers)):
        prompt = format_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if exact_match(text, gts):
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"    Eval {i+1}/{len(questions)} (acc: {correct/(i+1):.1%})", flush=True)
    return correct / len(questions)


def main():
    print("=" * 60)
    print("G2-CONTROL: REAL DATA BASELINE FOR CKA COMPARISON")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load TriviaQA
    print("\n[SETUP] Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=SEED)

    # Training data: REAL Q&A pairs (formatted as model would see them)
    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    train_answers_raw = [ds[i]["answer"]["value"] for i in range(TRAIN_SIZE)]

    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    probe_questions = [ds[i]["question"] for i in range(TRAIN_SIZE + EVAL_SIZE, TRAIN_SIZE + EVAL_SIZE + PROBE_SIZE)]

    # --- Gen 0: Base model ---
    print("\n[Gen 0] Base model representations...")
    model, tokenizer = load_base()

    gen0_reps = extract_reps(model, tokenizer, probe_questions, MONITOR_LAYERS)
    gen0_acc = evaluate_questions(model, tokenizer, eval_questions, eval_answers)
    print(f"  Gen0 accuracy: {gen0_acc:.1%}")

    # Format REAL training data (Q&A pairs with ground truth answers)
    print("\n[G2] Preparing REAL training data...")
    real_training_texts = []
    for q, a in zip(train_questions, train_answers_raw):
        prompt = format_prompt(tokenizer, q)
        full = prompt + a
        real_training_texts.append(full)

    # --- Train G2 adapter on REAL data ---
    print("[G2] Training LoRA on REAL data (same config as G1)...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, lora_config)

    def tok_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=256, padding="max_length")

    train_ds = Dataset.from_dict({"text": real_training_texts})
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "tmp"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=peft_model, args=args, train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    peft_model.eval()

    # --- Extract G2 representations ---
    print("\n[G2] Extracting representations after real-data fine-tuning...")
    g2_reps = extract_reps(peft_model, tokenizer, probe_questions, MONITOR_LAYERS)
    g2_acc = evaluate_questions(peft_model, tokenizer, eval_questions, eval_answers)
    print(f"  G2 accuracy: {g2_acc:.1%}")

    # --- Compare ---
    print("\n" + "=" * 60)
    print("G2-CONTROL RESULTS")
    print("=" * 60)

    # Load G1 results for comparison
    g1_results_path = Path(__file__).parent.parent / "outputs" / "m2_pilot" / "results.json"
    g1_cka_13 = None
    if g1_results_path.exists():
        with open(g1_results_path) as f:
            g1_data = json.load(f)
        for r in g1_data:
            if r["generation"] == 1 and "cka_vs_gen0" in r:
                g1_cka_13 = r["cka_vs_gen0"].get("13", {})
                break

    print(f"\n  {'Layer':<8} {'G2-Global':<14} {'G2-Factual':<14} {'G1-Factual (M2)':<16}")
    print(f"  {'-'*52}")
    for idx in MONITOR_LAYERS:
        cka_g = linear_cka(gen0_reps[idx]["global"], g2_reps[idx]["global"])
        cka_f = linear_cka(gen0_reps[idx]["f1"], g2_reps[idx]["f1"])
        g1_f = ""
        if g1_cka_13 and str(idx) in (g1_data[1].get("cka_vs_gen0", {}) if len(g1_data) > 1 else {}):
            g1_f = f"{g1_data[1]['cka_vs_gen0'][str(idx)]['f1']:.6f}"
        print(f"  {idx:<8} {cka_g:<14.6f} {cka_f:<14.6f} {g1_f}")

    # Key comparison at layer 13
    g2_cka_f13 = linear_cka(gen0_reps[13]["f1"], g2_reps[13]["f1"])
    g2_cka_g13 = linear_cka(gen0_reps[13]["global"], g2_reps[13]["global"])
    g1_f13 = g1_cka_13.get("f1", "N/A") if g1_cka_13 else "N/A"

    print(f"\n  KEY COMPARISON (Layer 13):")
    print(f"    G2 (Real data) CKA-Factual:  {g2_cka_f13:.6f}")
    print(f"    G1 (Synthetic) CKA-Factual:  {g1_f13}")
    print(f"    G2 CKA-Global:               {g2_cka_g13:.6f}")
    print(f"    Gen0 accuracy:               {gen0_acc:.1%}")
    print(f"    G2 accuracy:                 {g2_acc:.1%}")

    if isinstance(g1_f13, float):
        diff = g2_cka_f13 - g1_f13
        print(f"\n    Delta (G2 - G1): {diff:+.6f}")
        if diff > 0.005:
            print(f"    VERDICT: G2 > G1 -- Synthetic data causes MORE drift than real data")
            print(f"    --> CKA-Factual IS measuring something specific to synthetic recursion")
        elif diff < -0.005:
            print(f"    VERDICT: G1 > G2 -- Real data causes MORE drift")
            print(f"    --> CKA-Factual is NOT specific to synthetic data")
        else:
            print(f"    VERDICT: G1 ≈ G2 -- Drop is generic fine-tuning artifact")
            print(f"    --> CKA-Factual detects adaptation, not recursion specifically")

    # Save
    results = {
        "g2_cka_factual_layer13": g2_cka_f13,
        "g2_cka_global_layer13": g2_cka_g13,
        "g2_accuracy": g2_acc,
        "gen0_accuracy": gen0_acc,
        "g1_cka_factual_layer13": g1_f13,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
