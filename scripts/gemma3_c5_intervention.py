"""Gemma 3 Exposure Intervention: C5 random downsample at boundary (r=16).

Tests whether ~5% synthetic exposure reduction restores stability on Gemma 3,
the same way it does on Qwen r=256. This is the single most impactful
experiment to strengthen the cross-backbone generalization claim.

Gemma 3 boundary: r=16 (degradative, ~68.8% mean at Gen5).
Intervention: random downsample ~5% of training examples each generation.

Run on Athena:
  uv run python scripts/gemma3_c5_intervention.py

Expected:
  If retention improves from ~68% to ~80%+: exposure axis generalizes cross-backbone
  If no improvement: exposure sensitivity is Qwen-specific
"""
import sys, gc, json, time, re, argparse
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
from tqdm import tqdm

MODEL_NAME = "google/gemma-3-1b-it"
TRAIN_SIZE = 2000
EVAL_SIZE = 200
SEED = 15
GENERATIONS = 5
RANK = 16
LR = 1e-5
DOWNSAMPLE_FRACTION = 0.95  # Keep 95% (remove ~5%)
MASKS = [15, 42, 99]  # 3 independent random masks

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
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_prompt(tokenizer, q):
    msgs = [
        {"role": "user", "content": f"Answer the following trivia question in 5 words or less.\n\nQuestion: {q}"}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def evaluate_k0(model, tokenizer, k0_questions, k0_answers):
    results = []
    model.eval()
    for q, gts in zip(k0_questions, k0_answers):
        prompt = format_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append(exact_match(resp, gts))
    return results


def generate_synthetic(model, tokenizer, questions):
    responses = []
    model.eval()
    for q in tqdm(questions, desc="Generating"):
        prompt = format_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=50, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(resp)
    return responses


def downsample_dataset(questions, responses, rng):
    """Random downsample: keep DOWNSAMPLE_FRACTION of examples."""
    n = len(questions)
    n_keep = int(n * DOWNSAMPLE_FRACTION)
    indices = rng.choice(n, size=n_keep, replace=False)
    indices.sort()
    return [questions[i] for i in indices], [responses[i] for i in indices]


def train_adapter(model, tokenizer, questions, responses):
    """Train QLoRA adapter on synthetic data."""
    texts = []
    for q, r in zip(questions, responses):
        prompt = format_prompt(tokenizer, q)
        texts.append(prompt + r + tokenizer.eos_token)

    encodings = tokenizer(texts, truncation=True, max_length=256, padding="max_length")
    ds = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["input_ids"],
    })

    lora_config = LoraConfig(
        r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="/tmp/gemma3_c5_tmp",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=LR,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    return peft_model


def run_single_mask(mask_seed, train_questions, k0_questions, k0_answers, output_dir):
    """Run full 5-gen recursive loop with one downsample mask."""
    print(f"\n{'='*60}")
    print(f"  MASK SEED: {mask_seed}")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(mask_seed)
    results = {"mask_seed": mask_seed, "generations": []}

    # Gen 0: baseline
    model, tokenizer = load_base()
    k0_results = evaluate_k0(model, tokenizer, k0_questions, k0_answers)
    k0_correct = sum(k0_results)
    print(f"Gen 0: K0 = {k0_correct}/{len(k0_questions)}")
    results["k0_total"] = len(k0_questions)
    results["k0_correct_gen0"] = k0_correct

    # Generate initial synthetic
    responses = generate_synthetic(model, tokenizer, train_questions)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    for gen in range(1, GENERATIONS + 1):
        print(f"\n--- Generation {gen} (mask={mask_seed}) ---")

        # Downsample
        ds_questions, ds_responses = downsample_dataset(train_questions, responses, rng)
        print(f"  Training on {len(ds_questions)}/{len(train_questions)} examples "
              f"({len(ds_questions)/len(train_questions)*100:.1f}%)")

        # Load fresh base + train adapter
        model, tokenizer = load_base()
        peft_model = train_adapter(model, tokenizer, ds_questions, ds_responses)

        # Evaluate K0
        k0_results = evaluate_k0(peft_model, tokenizer, k0_questions, k0_answers)
        k0_correct = sum(k0_results)
        retention = k0_correct / results["k0_correct_gen0"] * 100
        print(f"  K0: {k0_correct}/{results['k0_correct_gen0']} ({retention:.1f}%)")

        results["generations"].append({
            "gen": gen,
            "k0_correct": k0_correct,
            "retention_pct": retention,
            "train_examples": len(ds_questions),
        })

        # Generate new synthetic (from FULL question set, not downsampled)
        responses = generate_synthetic(peft_model, tokenizer, train_questions)

        # Cleanup
        del peft_model, model
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    out_file = output_dir / f"gemma3_c5_mask{mask_seed}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", type=int, nargs="+", default=MASKS,
                        help="Random seeds for downsampling masks")
    args = parser.parse_args()

    output_dir = Path("outputs/gemma3_c5_intervention")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load TriviaQA
    print("Loading TriviaQA...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(ds))

    train_indices = indices[:TRAIN_SIZE]
    eval_indices = indices[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]

    train_questions = [ds[int(i)]["question"] for i in train_indices]
    eval_questions = [ds[int(i)]["question"] for i in eval_indices]
    eval_answers = [ds[int(i)]["answer"]["aliases"] for i in eval_indices]

    # Establish K0 (which facts the base model knows)
    print("Establishing K0...")
    model, tokenizer = load_base()
    k0_results = evaluate_k0(model, tokenizer, eval_questions, eval_answers)
    k0_mask = [i for i, correct in enumerate(k0_results) if correct]
    k0_questions = [eval_questions[i] for i in k0_mask]
    k0_answers = [eval_answers[i] for i in k0_mask]
    print(f"K0 size: {len(k0_questions)} / {EVAL_SIZE}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Run each mask
    all_results = []
    for mask_seed in args.masks:
        result = run_single_mask(mask_seed, train_questions, k0_questions, k0_answers, output_dir)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Gemma 3 C5 Intervention (r=16, 5% downsample)")
    print("=" * 60)
    for r in all_results:
        gen5 = r["generations"][-1]
        print(f"  Mask {r['mask_seed']}: Gen5 = {gen5['k0_correct']}/{r['k0_correct_gen0']} "
              f"({gen5['retention_pct']:.1f}%)")

    # Compare with baseline (Gemma 3 r=16 without intervention: ~68.8%)
    mean_ret = np.mean([r["generations"][-1]["retention_pct"] for r in all_results])
    print(f"\n  Mean retention (C5): {mean_ret:.1f}%")
    print(f"  Baseline (no intervention): ~68.8%")
    print(f"  Delta: +{mean_ret - 68.8:.1f}pp")


if __name__ == "__main__":
    main()
