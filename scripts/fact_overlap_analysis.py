"""Extract per-item k0_results from saved synthetic data and compute fact-overlap.

For FFT: reloads model, trains on saved synthetics, evals K0 per-item.
For QLoRA: same.
Then computes Jaccard overlap between methods and across seeds.

This requires GPU but is fast (~5min per seed, inference only on saved checkpoints).
Actually: we don't need to retrain. We need to eval the SAME way the original script did.
But we don't have saved model checkpoints — only synthetic data.

ALTERNATIVE APPROACH: Since FFT/QLoRA are deterministic (same seed, same data, same LR),
re-running the exact training + eval pipeline for Gen1 only gives us k0_results.

Run: uv run python scripts/fact_overlap_analysis.py
"""
import sys, gc, json, time
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
import re

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_SIZE = 2000
EVAL_SIZE = 200
SEEDS = [15, 137, 256]

NUMBER_WORDS = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}

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

def format_prompt(tokenizer, q):
    msgs = [{"role": "system", "content": "Answer the following question in 5 words or less."},
            {"role": "user", "content": q}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def evaluate_k0_detailed(model, tokenizer, k0_questions, k0_answers):
    results = []
    for q, gts in zip(k0_questions, k0_answers):
        prompt = format_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append(exact_match(text, gts))
    return results

def train_fft_gen1(seed, train_synthetic, model_name):
    """Train FFT for 1 gen and return model for eval."""
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=256, padding="max_length")
    train_ds = Dataset.from_dict({"text": train_synthetic})
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")

    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir="/tmp/fact_overlap", num_train_epochs=2,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=1e-6, bf16=True, logging_steps=9999,
        save_strategy="no", report_to="none", seed=seed + 1,
        gradient_checkpointing=True, optim="paged_adamw_8bit"),
        train_dataset=train_ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()
    model.eval()
    del trainer, train_ds; gc.collect(); torch.cuda.empty_cache()
    return model, tok

def train_qlora_gen1(seed, train_synthetic, model_name):
    """Train QLoRA for 1 gen and return model for eval."""
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb,
                                                  device_map="auto", torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    torch.manual_seed(seed + 1)
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=256, padding="max_length")
    train_ds = Dataset.from_dict({"text": train_synthetic})
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")

    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir="/tmp/fact_overlap", num_train_epochs=2,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=1e-5, bf16=True, logging_steps=9999,
        save_strategy="no", report_to="none", seed=seed + 1),
        train_dataset=train_ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()
    model.eval()
    del trainer, train_ds; gc.collect(); torch.cuda.empty_cache()
    return model, tok


def main():
    output_dir = Path(__file__).parent.parent / "outputs" / "fact_overlap"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "fact_overlap_results.json"
    if result_path.exists():
        print("Already done. Results:")
        results = json.load(open(result_path))
        print_summary(results)
        return

    # Load data (same as other scripts)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)
    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    # K0 indices (same as sweep)
    sweep_k0 = Path(__file__).parent.parent / "outputs" / "fft_lr_sweep" / "k0_indices.json"
    k0_indices = json.load(open(sweep_k0))
    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]
    print(f"K0 size: {len(k0_indices)}")

    results = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # Load synthetic gen0 for this seed
        syn_path = Path(__file__).parent.parent / "outputs" / "fft_drift_replicate" / f"synthetic_gen0_seed{seed}.json"
        if not syn_path.exists():
            # Use sweep synthetic for seed 15
            syn_path = Path(__file__).parent.parent / "outputs" / "fft_lr_sweep" / "synthetic_gen0.json"
        synthetic = json.load(open(syn_path))

        # Train FFT Gen1 and eval
        print(f"  Training FFT Gen1 seed={seed}...")
        model, tok = train_fft_gen1(seed, synthetic, MODEL_NAME)
        fft_k0 = evaluate_k0_detailed(model, tok, k0_questions, k0_answers)
        fft_lost = [i for i, r in enumerate(fft_k0) if not r]
        print(f"  FFT: {sum(fft_k0)}/{len(fft_k0)} retained, lost: {fft_lost}")
        del model; gc.collect(); torch.cuda.empty_cache()

        # Train QLoRA Gen1 and eval
        print(f"  Training QLoRA Gen1 seed={seed}...")
        model, tok = train_qlora_gen1(seed, synthetic, MODEL_NAME)
        qlora_k0 = evaluate_k0_detailed(model, tok, k0_questions, k0_answers)
        qlora_lost = [i for i, r in enumerate(qlora_k0) if not r]
        print(f"  QLoRA: {sum(qlora_k0)}/{len(qlora_k0)} retained, lost: {qlora_lost}")
        del model; gc.collect(); torch.cuda.empty_cache()

        # Compute overlap
        shared = set(fft_lost) & set(qlora_lost)
        union = set(fft_lost) | set(qlora_lost)
        jaccard = len(shared) / len(union) if union else 1.0

        results[f"seed_{seed}"] = {
            "fft_lost": fft_lost,
            "qlora_lost": qlora_lost,
            "fft_retained": sum(fft_k0),
            "qlora_retained": sum(qlora_k0),
            "shared_lost": list(shared),
            "jaccard": jaccard,
        }

    # Cross-seed FFT overlap
    fft_sets = [set(results[f"seed_{s}"]["fft_lost"]) for s in SEEDS]
    cross_fft_jaccard_01 = len(fft_sets[0] & fft_sets[1]) / len(fft_sets[0] | fft_sets[1]) if (fft_sets[0] | fft_sets[1]) else 1
    cross_fft_jaccard_02 = len(fft_sets[0] & fft_sets[2]) / len(fft_sets[0] | fft_sets[2]) if (fft_sets[0] | fft_sets[2]) else 1
    cross_fft_jaccard_12 = len(fft_sets[1] & fft_sets[2]) / len(fft_sets[1] | fft_sets[2]) if (fft_sets[1] | fft_sets[2]) else 1

    results["cross_seed_fft"] = {
        "jaccard_15_137": cross_fft_jaccard_01,
        "jaccard_15_256": cross_fft_jaccard_02,
        "jaccard_137_256": cross_fft_jaccard_12,
        "intersection_all": list(fft_sets[0] & fft_sets[1] & fft_sets[2]),
    }

    json.dump(results, open(result_path, "w"), indent=2)
    print_summary(results)


def print_summary(results):
    print("\n" + "=" * 70)
    print("FACT-OVERLAP ANALYSIS SUMMARY")
    print("=" * 70)

    for seed in SEEDS:
        key = f"seed_{seed}"
        if key not in results:
            continue
        r = results[key]
        print(f"\n  Seed {seed}:")
        print(f"    FFT lost ({len(r['fft_lost'])}): {r['fft_lost']}")
        print(f"    QLoRA lost ({len(r['qlora_lost'])}): {r['qlora_lost']}")
        print(f"    Shared: {r['shared_lost']}")
        print(f"    Jaccard(FFT, QLoRA): {r['jaccard']:.3f}")

    if "cross_seed_fft" in results:
        c = results["cross_seed_fft"]
        print(f"\n  Cross-seed FFT overlap:")
        print(f"    Jaccard(seed15, seed137): {c['jaccard_15_137']:.3f}")
        print(f"    Jaccard(seed15, seed256): {c['jaccard_15_256']:.3f}")
        print(f"    Jaccard(seed137, seed256): {c['jaccard_137_256']:.3f}")
        print(f"    Facts lost in ALL 3 seeds: {c['intersection_all']}")

    print("\n  INTERPRETATION:")
    print("    High cross-seed Jaccard = same fragile facts lost deterministically")
    print("    Low FFT-vs-QLoRA Jaccard = methods perturb different fact subspaces")


if __name__ == "__main__":
    main()
