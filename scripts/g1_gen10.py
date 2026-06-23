"""G1-Gen10: Extend recursive synthetic training to 10 generations.

Answers: "Does collapse eventually appear after more iterations?"

Uses same protocol as M2 (data-only recursion, new adapter each gen).
Resumes from M2 checkpoints if available.

Run: uv run python scripts/g1_gen10.py [--seed 15]
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
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_SIZE = 2000
EVAL_SIZE = 200
NUM_GENERATIONS = 10

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
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append(exact_match(text, gts))
        if (i + 1) % 50 == 0:
            print(f"      Eval {i+1}/{len(questions)} (acc: {sum(results)/(i+1):.1%})", flush=True)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=15)
    args = parser.parse_args()
    seed = args.seed

    output_dir = Path(__file__).parent.parent / "outputs" / f"g1_gen10_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"G1-GEN10: 10 GENERATIONS RECURSIVE SYNTHETIC (seed={seed})")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("\n[SETUP] Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)  # Fixed split

    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    # Check for resume
    results_path = output_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        last_gen = max(r["generation"] for r in results)
        syn_path = output_dir / f"synthetic_gen{last_gen}.json"
        if syn_path.exists():
            with open(syn_path) as f:
                prev_synthetic = json.load(f)
            start_gen = last_gen + 1
            k0_indices = results[0]["k0_indices"]
            print(f"  Resuming from Gen {start_gen}")
        else:
            start_gen = 0
            results = []
    else:
        start_gen = 0
        results = []

    # Gen 0
    if start_gen == 0:
        print(f"\n[Gen 0] Base model...")
        model, tokenizer = load_base()

        gen0_results = evaluate_questions(model, tokenizer, eval_questions, eval_answers)
        k0_indices = [i for i, r in enumerate(gen0_results) if r]
        acc = sum(gen0_results) / len(gen0_results)
        print(f"    Accuracy: {acc:.1%}, K0={len(k0_indices)}")

        print("    Generating synthetic...")
        torch.manual_seed(seed)
        prev_synthetic = generate_synthetic(model, tokenizer, train_questions)

        with open(output_dir / "synthetic_gen0.json", "w") as f:
            json.dump(prev_synthetic, f)

        results = [{"generation": 0, "accuracy": acc, "k0_size": len(k0_indices),
                    "k0_indices": k0_indices, "retention": len(k0_indices), "transitions": None}]
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        del model
        gc.collect(); torch.cuda.empty_cache()
        start_gen = 1

    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]

    # Gen 1-10
    for gen in range(start_gen, NUM_GENERATIONS + 1):
        print(f"\n[Gen {gen}]")
        model, tokenizer = load_base()

        # Train
        print("    Training...")
        torch.manual_seed(seed + gen)
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                 target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        model.enable_input_require_grads()
        peft_model = get_peft_model(model, lora_config)

        def tok_fn(ex):
            return tokenizer(ex["text"], truncation=True, max_length=256, padding="max_length")
        train_ds = Dataset.from_dict({"text": prev_synthetic})
        train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
        train_ds.set_format("torch")

        trainer = Trainer(
            model=peft_model,
            args=TrainingArguments(
                output_dir=str(output_dir / "tmp"), num_train_epochs=2,
                per_device_train_batch_size=2, gradient_accumulation_steps=8,
                learning_rate=1e-5, bf16=True, logging_steps=100,
                save_strategy="no", report_to="none", seed=seed + gen,
            ),
            train_dataset=train_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        peft_model.eval()

        # Evaluate K0
        print("    Evaluating K0...")
        k0_res = evaluate_questions(peft_model, tokenizer, k0_questions, k0_answers)
        ret = sum(k0_res)
        print(f"    K0 retention: {ret}/{len(k0_indices)} = {ret/len(k0_indices):.1%}")

        # Evaluate all
        print("    Evaluating all...")
        all_res = evaluate_questions(peft_model, tokenizer, eval_questions, eval_answers)
        acc = sum(all_res) / len(all_res)
        print(f"    Global: {acc:.1%}")

        # Transitions vs previous
        if len(results) > 0:
            prev_res = results[-1].get("all_results", [True if i in k0_indices else False for i in range(EVAL_SIZE)])
            cc = sum(1 for p, c in zip(prev_res, all_res) if p and c)
            cw = sum(1 for p, c in zip(prev_res, all_res) if p and not c)
            wc = sum(1 for p, c in zip(prev_res, all_res) if not p and c)
            ww = sum(1 for p, c in zip(prev_res, all_res) if not p and not c)
            trans = {"C->C": cc, "C->W": cw, "W->C": wc, "W->W": ww}
            print(f"    Transitions: C->C={cc} C->W={cw} W->C={wc} W->W={ww}")
        else:
            trans = None

        # Generate next synthetic
        print("    Generating synthetic...")
        torch.manual_seed(seed + gen + 100)
        new_synthetic = generate_synthetic(peft_model, tokenizer, train_questions)

        with open(output_dir / f"synthetic_gen{gen}.json", "w") as f:
            json.dump(new_synthetic, f)

        results.append({"generation": gen, "accuracy": acc, "retention": ret,
                        "transitions": trans, "all_results": all_res})
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        prev_synthetic = new_synthetic
        del peft_model, trainer, model
        gc.collect(); torch.cuda.empty_cache()

    # Final report
    print("\n" + "=" * 60)
    print("G1-GEN10 FINAL REPORT")
    print("=" * 60)
    print(f"\n  {'Gen':<6} {'Accuracy':<12} {'K0 Retention':<14} {'C->W':<6} {'W->C'}")
    print(f"  {'-'*44}")
    for r in results:
        t = r.get("transitions")
        cw = t["C->W"] if t else "-"
        wc = t["W->C"] if t else "-"
        ret_pct = f"{r['retention']}/{len(k0_indices)}" if "retention" in r else "-"
        print(f"  {r['generation']:<6} {r['accuracy']:<12.1%} {ret_pct:<14} {cw:<6} {wc}")


if __name__ == "__main__":
    main()
