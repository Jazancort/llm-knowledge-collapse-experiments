"""M2 Audit: Inspect Gen0 accuracy errors to diagnose metric validity.

Classifies errors into:
- Format: correct fact, wrong format (e.g. "The answer is Paris" vs "Paris")
- Partial: overlapping but not matching (e.g. "Washington" vs "George Washington")
- Factual: genuinely wrong answer

Also computes Retention Accuracy: tracking only questions correct at Gen0.

Run: uv run python scripts/m2_audit.py
"""

import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 15
TRAIN_SIZE = 2000
EVAL_SIZE = 200
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "m2_audit"

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


def classify_error(pred, ground_truths):
    """Classify why exact_match failed."""
    pred_norm = normalize_answer(pred)
    for gt in ground_truths:
        gt_norm = normalize_answer(gt)
        # Check if GT appears anywhere in prediction (format issue)
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return "format"  # Should have matched — bug in exact_match
        # Check partial overlap (shared significant words)
        pred_words = set(pred_norm.split())
        gt_words = set(gt_norm.split())
        overlap = pred_words & gt_words
        if len(overlap) >= 1 and len(gt_words) > 0:
            if len(overlap) / len(gt_words) >= 0.5:
                return "partial"
    return "factual"


def main():
    print("=" * 60)
    print("M2 AUDIT: ACCURACY ERROR ANALYSIS")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n[1/3] Loading TriviaQA eval set...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=SEED)

    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    # Load model
    print("\n[2/3] Loading model and generating answers...")
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
    model.eval()

    # Generate all answers deterministically
    predictions = []
    for idx, q in enumerate(eval_questions):
        msgs = [
            {"role": "system", "content": "Answer the following question in 5 words or less."},
            {"role": "user", "content": q},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        predictions.append(text)
        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{EVAL_SIZE}...", flush=True)

    # Classify
    print(f"\n[3/3] Classifying errors...")
    correct_indices = []
    errors = {"format": [], "partial": [], "factual": []}

    for i, (pred, gts) in enumerate(zip(predictions, eval_answers)):
        if exact_match(pred, gts):
            correct_indices.append(i)
        else:
            err_type = classify_error(pred, gts)
            errors[err_type].append({
                "idx": i,
                "question": eval_questions[i],
                "prediction": pred,
                "ground_truth": gts[:3],
            })

    total = len(eval_questions)
    n_correct = len(correct_indices)
    n_format = len(errors["format"])
    n_partial = len(errors["partial"])
    n_factual = len(errors["factual"])

    print(f"\n{'=' * 60}")
    print("AUDIT RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  Total: {total}")
    print(f"  Correct (EM):     {n_correct} ({n_correct/total:.1%})")
    print(f"  Error - Format:   {n_format} ({n_format/total:.1%})")
    print(f"  Error - Partial:  {n_partial} ({n_partial/total:.1%})")
    print(f"  Error - Factual:  {n_factual} ({n_factual/total:.1%})")
    print(f"\n  'True' accuracy (correct + format + partial): {(n_correct+n_format+n_partial)/total:.1%}")
    print(f"  Strict EM accuracy: {n_correct/total:.1%}")

    # Show sample errors
    print(f"\n  --- SAMPLE FORMAT ERRORS (first 10) ---")
    for e in errors["format"][:10]:
        print(f"    Q: {e['question'][:60].encode('ascii', 'replace').decode()}")
        print(f"    P: {e['prediction'][:40].encode('ascii', 'replace').decode()}  |  GT: {e['ground_truth'][0][:30].encode('ascii', 'replace').decode()}")
        print()

    print(f"  --- SAMPLE PARTIAL ERRORS (first 10) ---")
    for e in errors["partial"][:10]:
        print(f"    Q: {e['question'][:60].encode('ascii', 'replace').decode()}")
        print(f"    P: {e['prediction'][:40].encode('ascii', 'replace').decode()}  |  GT: {e['ground_truth'][0][:30].encode('ascii', 'replace').decode()}")
        print()

    print(f"  --- SAMPLE FACTUAL ERRORS (first 10) ---")
    for e in errors["factual"][:10]:
        print(f"    Q: {e['question'][:60].encode('ascii', 'replace').decode()}")
        print(f"    P: {e['prediction'][:40].encode('ascii', 'replace').decode()}  |  GT: {e['ground_truth'][0][:30].encode('ascii', 'replace').decode()}")
        print()

    # Save
    audit_data = {
        "total": total,
        "correct": n_correct,
        "format_errors": n_format,
        "partial_errors": n_partial,
        "factual_errors": n_factual,
        "correct_indices": correct_indices,
        "errors": errors,
    }
    with open(OUTPUT_DIR / "audit_gen0.json", "w") as f:
        json.dump(audit_data, f, indent=2)

    print(f"\n  Saved to {OUTPUT_DIR / 'audit_gen0.json'}")
    print(f"  correct_indices saved ({n_correct} items) for Retention Accuracy in M3")


if __name__ == "__main__":
    main()
