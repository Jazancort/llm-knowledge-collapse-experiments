"""M2 Multi-Seed: Run retention analysis with seeds 2 and 3.

Replicates the full M2 retention pipeline (Gen0-3) with different seeds
to verify if the stabilization pattern is reproducible.

Usage:
  uv run python scripts/m2_multiseed.py --seed 137
  uv run python scripts/m2_multiseed.py --seed 256

Each seed gets its own output directory.
"""

import sys
import gc
import json
import re
import argparse
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

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_SIZE = 2000
EVAL_SIZE = 200
NUM_GENERATIONS = 3

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


def evaluate_questions(model, tokenizer, questions, answers):
    results = []
    for i, (q, gts) in enumerate(zip(questions, answers)):
        prompt = format_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        results.append(exact_match(text, gts))
        if (i + 1) % 50 == 0:
            print(f"      Eval {i+1}/{len(questions)}...", flush=True)
    return results


def generate_synthetic(model, tokenizer, questions):
    synthetic = []
    for i in range(0, len(questions), 8):
        batch = questions[i:i+8]
        prompts = [format_prompt(tokenizer, q) for q in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, temperature=0.7, do_sample=True, top_p=0.9)
        for seq in out:
            synthetic.append(tokenizer.decode(seq, skip_special_tokens=False))
        if (i + 8) % 500 == 0:
            print(f"      Gen {min(i+8, len(questions))}/{len(questions)}...", flush=True)
    return synthetic


def train_adapter(model, tokenizer, synthetic_data, output_dir, seed):
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, lora_config)

    def tok_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=256, padding="max_length")

    ds = Dataset.from_dict({"text": synthetic_data})
    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")

    args = TrainingArguments(
        output_dir=str(output_dir / "tmp"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(
        model=peft_model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    peft_model.eval()
    return peft_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = args.seed

    output_dir = Path(__file__).parent.parent / "outputs" / f"m2_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"M2 MULTI-SEED: Seed={seed}")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data (same split as seed 15 — eval set is fixed)
    print("\n[SETUP] Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)  # SAME shuffle as original — eval set must be identical

    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    # --- Gen 0 ---
    print("\n[Gen 0] Base model...")
    model, tokenizer = load_base()

    print("    Evaluating...")
    gen0_results = evaluate_questions(model, tokenizer, eval_questions, eval_answers)
    k0_indices = [i for i, r in enumerate(gen0_results) if r]
    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]
    print(f"    Accuracy: {sum(gen0_results)}/{len(gen0_results)} = {sum(gen0_results)/len(gen0_results):.1%}")
    print(f"    K0 size: {len(k0_indices)}")

    print("    Generating synthetic...")
    # Use THIS seed for generation randomness
    torch.manual_seed(seed)
    synthetic = generate_synthetic(model, tokenizer, train_questions)

    del model
    gc.collect(); torch.cuda.empty_cache()

    # Track
    retention = {0: len(k0_indices)}
    transitions = {}
    prev_synthetic = synthetic
    prev_all_results = gen0_results

    # --- Gen 1-3 ---
    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n[Gen {gen}]")
        model, tokenizer = load_base()

        print("    Training adapter...")
        torch.manual_seed(seed + gen)  # Different init per gen but reproducible
        peft_model = train_adapter(model, tokenizer, prev_synthetic, output_dir, seed + gen)

        print("    Evaluating K0...")
        k0_res = evaluate_questions(peft_model, tokenizer, k0_questions, k0_answers)
        retention[gen] = sum(k0_res)
        print(f"    K0 retention: {sum(k0_res)}/{len(k0_indices)} = {sum(k0_res)/len(k0_indices):.1%}")

        print("    Evaluating all...")
        all_res = evaluate_questions(peft_model, tokenizer, eval_questions, eval_answers)
        print(f"    Global: {sum(all_res)}/{len(all_res)} = {sum(all_res)/len(all_res):.1%}")

        # Transitions
        cc = sum(1 for p, c in zip(prev_all_results, all_res) if p and c)
        cw = sum(1 for p, c in zip(prev_all_results, all_res) if p and not c)
        wc = sum(1 for p, c in zip(prev_all_results, all_res) if not p and c)
        ww = sum(1 for p, c in zip(prev_all_results, all_res) if not p and not c)
        transitions[gen] = {"C->C": cc, "C->W": cw, "W->C": wc, "W->W": ww}
        print(f"    Transitions: C->C={cc} C->W={cw} W->C={wc} W->W={ww}")

        print("    Generating synthetic for next gen...")
        torch.manual_seed(seed + gen + 100)
        new_synthetic = generate_synthetic(peft_model, tokenizer, train_questions)

        prev_synthetic = new_synthetic
        prev_all_results = all_res
        del peft_model, model
        gc.collect(); torch.cuda.empty_cache()

    # --- Report ---
    print("\n" + "=" * 60)
    print(f"SEED {seed} FINAL REPORT")
    print("=" * 60)
    print(f"\n  K0 Retention (N={len(k0_indices)}):")
    for gen in range(NUM_GENERATIONS + 1):
        n = retention[gen]
        print(f"    Gen {gen}: {n}/{len(k0_indices)} = {n/len(k0_indices):.1%}")

    print(f"\n  Transitions:")
    for gen in range(1, NUM_GENERATIONS + 1):
        t = transitions[gen]
        print(f"    Gen {gen}: C->C={t['C->C']} C->W={t['C->W']} W->C={t['W->C']} W->W={t['W->W']}")

    # Save
    results = {"seed": seed, "k0_size": len(k0_indices), "retention": retention, "transitions": transitions}
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
