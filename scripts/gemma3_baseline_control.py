"""Gemma 3 Baseline Control: Same script as C5 intervention, but WITHOUT downsample.

Purpose: Internal control to determine whether the 93.5% retention observed in
the C5 intervention is due to the 5% downsample, or due to prompt/dataset
differences from the original sprint2_gemma.py baseline.

If this gives ~93%: the effect is from prompt/dataset, NOT from downsample.
If this gives ~68%: the C5 effect is real.

Run on Athena:
  uv run python scripts/gemma3_baseline_control.py
"""
import sys, gc, json, time, re
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


def defrag():
    gc.collect()
    torch.cuda.empty_cache()


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
    # SAME prompt as gemma3_c5_intervention.py
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
    batch_size = 16
    tokenizer.padding_side = "left"
    for i in tqdm(range(0, len(questions), batch_size), desc="Generating"):
        batch_q = questions[i:i + batch_size]
        prompts = [format_prompt(tokenizer, q) for q in batch_q]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=50, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, seq in enumerate(out):
            resp = tokenizer.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            responses.append(resp)
    tokenizer.padding_side = "right"
    return responses


def train_adapter(model, tokenizer, questions, responses, gen):
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
        output_dir="/tmp/gemma3_ctrl_tmp",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=LR,
        bf16=True,
        logging_steps=9999,
        save_strategy="no",
        report_to="none",
        seed=SEED + gen,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    peft_model.eval()
    del trainer, ds
    defrag()
    return peft_model


def main():
    output_dir = Path("outputs/gemma3_baseline_control")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "gemma3_baseline_no_downsample.json"

    # Load TriviaQA — SAME split as C5 intervention
    print("Loading TriviaQA...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(ds))

    train_indices = indices[:TRAIN_SIZE]
    eval_indices = indices[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]

    train_questions = [ds[int(i)]["question"] for i in train_indices]
    eval_questions = [ds[int(i)]["question"] for i in eval_indices]
    eval_answers = [ds[int(i)]["answer"]["aliases"] for i in eval_indices]

    # Establish K0
    print("Establishing K0...")
    model, tokenizer = load_base()
    k0_results = evaluate_k0(model, tokenizer, eval_questions, eval_answers)
    k0_mask = [i for i, correct in enumerate(k0_results) if correct]
    k0_questions = [eval_questions[i] for i in k0_mask]
    k0_answers = [eval_answers[i] for i in k0_mask]
    k0_total = len(k0_questions)
    print(f"K0 size: {k0_total} / {EVAL_SIZE}")

    # Gen 0
    k0_eval = evaluate_k0(model, tokenizer, k0_questions, k0_answers)
    k0_correct = sum(k0_eval)
    print(f"Gen 0: K0 = {k0_correct}/{k0_total}")

    # Generate initial synthetic — NO DOWNSAMPLE (100% of 2000)
    responses = generate_synthetic(model, tokenizer, train_questions)
    del model
    defrag()

    gen_results = []

    for gen in range(1, GENERATIONS + 1):
        t0 = time.time()
        print(f"\n--- Generation {gen} (NO downsample, 100% data) ---")
        print(f"  Training on {len(train_questions)}/{len(train_questions)} examples (100%)")

        # Load fresh base + train adapter on ALL data
        model, tokenizer = load_base()
        peft_model = train_adapter(model, tokenizer, train_questions, responses, gen)

        # Evaluate K0
        k0_eval = evaluate_k0(peft_model, tokenizer, k0_questions, k0_answers)
        k0_correct_gen = sum(k0_eval)
        retention = k0_correct_gen / k0_correct * 100
        elapsed = time.time() - t0
        print(f"  K0: {k0_correct_gen}/{k0_correct} ({retention:.1f}%) [{elapsed:.0f}s]")

        gen_results.append({
            "gen": gen,
            "k0_correct": k0_correct_gen,
            "k0_total": k0_correct,
            "retention_pct": round(retention, 2),
            "train_examples": len(train_questions),
            "time_seconds": round(elapsed, 1),
        })

        # Save incrementally
        result_data = {
            "condition": "baseline_no_downsample",
            "model": MODEL_NAME,
            "rank": RANK,
            "downsample_fraction": 1.0,
            "k0_size": k0_total,
            "k0_correct_gen0": k0_correct,
            "generations": gen_results,
        }
        result_path.write_text(json.dumps(result_data, indent=2))
        print(f"  Saved: {result_path}")

        # Generate new synthetic
        responses = generate_synthetic(peft_model, tokenizer, train_questions)

        del peft_model, model
        defrag()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Gemma 3 Baseline Control (r=16, NO downsample)")
    print("=" * 60)
    for r in gen_results:
        print(f"  Gen {r['gen']}: {r['k0_correct']}/{r['k0_total']} ({r['retention_pct']:.1f}%)")
    print(f"\n  If ~93%: C5 effect is from prompt/dataset, not downsample")
    print(f"  If ~68%: C5 effect is REAL")


if __name__ == "__main__":
    main()
