"""M2 Retention: Compute retention accuracy over K0 (Gen0 correct set).

Uses existing M2 checkpoints — NO new training or generation needed.
Loads model + each generation's adapter, evaluates ONLY the 79 questions
that Gen0 answered correctly, and tracks C->W / W->C transitions.

Run: uv run python scripts/m2_retention.py
"""

import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 15
TRAIN_SIZE = 2000
EVAL_SIZE = 200
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "m2_pilot"
AUDIT_DIR = Path(__file__).parent.parent / "outputs" / "m2_audit"

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
    model.eval()
    return model, tokenizer


def evaluate_questions(model, tokenizer, questions, answers):
    """Returns list of booleans (correct/incorrect) per question."""
    results = []
    for q, gts in zip(questions, answers):
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
        results.append(exact_match(text, gts))
    return results


def train_adapter_on_synthetic(model, tokenizer, synthetic_data, save_path=None):
    """Train a fresh LoRA adapter on synthetic data. Optionally save adapter."""
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, lora_config)

    def tok_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

    ds = Dataset.from_dict({"text": synthetic_data})
    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "retention_tmp"),
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
        model=peft_model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    peft_model.eval()

    if save_path:
        peft_model.save_pretrained(save_path)
        print(f"    Adapter saved to {save_path}")

    return peft_model


def main():
    print("=" * 60)
    print("M2 RETENTION ACCURACY ANALYSIS")
    print("=" * 60)

    # Load audit data to get K0
    audit_path = AUDIT_DIR / "audit_gen0.json"
    if not audit_path.exists():
        print("ERROR: Run m2_audit.py first to generate audit_gen0.json")
        return

    with open(audit_path) as f:
        audit = json.load(f)

    k0_indices = audit["correct_indices"]
    print(f"\n  K0 size: {len(k0_indices)} questions (Gen0 correct)")

    # Freeze K0 to file (never recalculate)
    k0_frozen_path = OUTPUT_DIR / "k0_indices_frozen.json"
    if not k0_frozen_path.exists():
        with open(k0_frozen_path, "w") as f:
            json.dump(k0_indices, f)
        print(f"  K0 frozen to {k0_frozen_path}")
    else:
        with open(k0_frozen_path) as f:
            k0_indices = json.load(f)
        print(f"  K0 loaded from frozen file ({len(k0_indices)} items)")

    # Load TriviaQA eval set
    print("  Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=SEED)

    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]

    # K0 subset
    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]

    # Full eval for transition matrix
    all_questions = eval_questions
    all_answers = eval_answers

    # Track results per generation
    gen_results_k0 = {}  # gen -> list of bools for K0
    gen_results_all = {}  # gen -> list of bools for all 200

    # --- Gen 0 ---
    print(f"\n[Gen 0] Evaluating (base model)...")
    model, tokenizer = load_base()

    gen_results_k0[0] = [True] * len(k0_indices)  # By definition all correct
    gen_results_all[0] = evaluate_questions(model, tokenizer, all_questions, all_answers)
    print(f"  K0 retention: {sum(gen_results_k0[0])}/{len(k0_indices)} = 100.0%")
    print(f"  Global acc: {sum(gen_results_all[0])}/{len(all_questions)} = {sum(gen_results_all[0])/len(all_questions):.1%}")

    # Generate synthetic for Gen1
    syn0_path = OUTPUT_DIR / "synthetic_gen0.json"
    if syn0_path.exists():
        print("  Loading existing synthetic_gen0...")
        with open(syn0_path) as f:
            synthetic = json.load(f)
    else:
        print("  Generating synthetic data for Gen1...")
        synthetic = []
        for i in range(0, len(train_questions), 8):
            batch = train_questions[i:i+8]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": "Answer the following question in 5 words or less."},
                 {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True) for q in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30, temperature=0.7, do_sample=True, top_p=0.9)
            for seq in out:
                synthetic.append(tokenizer.decode(seq, skip_special_tokens=False))
            if (i + 8) % 500 == 0:
                print(f"    {min(i+8, len(train_questions))}/{len(train_questions)}...", flush=True)
        with open(syn0_path, "w") as f:
            json.dump(synthetic, f)

    del model
    import gc; gc.collect(); torch.cuda.empty_cache()

    # --- Gen 1-3 ---
    prev_synthetic = synthetic
    for gen in range(1, 4):
        print(f"\n[Gen {gen}] Training + evaluating...")
        model, tokenizer = load_base()

        adapter_path = OUTPUT_DIR / f"adapter_gen{gen}"
        synthetic_path = OUTPUT_DIR / f"synthetic_gen{gen-1}.json"

        # Try loading existing adapter
        if adapter_path.exists():
            print(f"  Loading saved adapter from {adapter_path}...")
            from peft import PeftModel as PM
            peft_model = PM.from_pretrained(model, str(adapter_path))
            peft_model.eval()
        else:
            # Need synthetic data from previous gen
            if synthetic_path.exists() and prev_synthetic is None:
                with open(synthetic_path) as f:
                    prev_synthetic = json.load(f)

            peft_model = train_adapter_on_synthetic(
                model, tokenizer, prev_synthetic, save_path=str(adapter_path)
            )

        # Evaluate K0
        print(f"  Evaluating K0 ({len(k0_indices)} questions)...")
        k0_res = evaluate_questions(peft_model, tokenizer, k0_questions, k0_answers)
        gen_results_k0[gen] = k0_res
        retention = sum(k0_res) / len(k0_res)
        print(f"  K0 retention: {sum(k0_res)}/{len(k0_res)} = {retention:.1%}")

        # Evaluate all for transition
        print(f"  Evaluating all 200...")
        all_res = evaluate_questions(peft_model, tokenizer, all_questions, all_answers)
        gen_results_all[gen] = all_res
        print(f"  Global acc: {sum(all_res)}/{len(all_res)} = {sum(all_res)/len(all_res):.1%}")

        # Generate next synthetic (and save it)
        print(f"  Generating synthetic for Gen{gen+1}...")
        new_synthetic = []
        for i in range(0, len(train_questions), 8):
            batch = train_questions[i:i+8]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": "Answer the following question in 5 words or less."},
                 {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True) for q in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = peft_model.generate(**inputs, max_new_tokens=30, temperature=0.7, do_sample=True, top_p=0.9)
            for seq in out:
                new_synthetic.append(tokenizer.decode(seq, skip_special_tokens=False))

        # Save synthetic
        syn_save_path = OUTPUT_DIR / f"synthetic_gen{gen}.json"
        with open(syn_save_path, "w") as f:
            json.dump(new_synthetic, f)

        prev_synthetic = new_synthetic
        del peft_model, model
        gc.collect(); torch.cuda.empty_cache()

    # --- Report ---
    print("\n" + "=" * 60)
    print("RETENTION ACCURACY REPORT")
    print("=" * 60)

    print(f"\n  K0 Retention (N={len(k0_indices)}):")
    print(f"  {'Gen':<6} {'Retained':<12} {'Retention %'}")
    print(f"  {'-'*32}")
    for gen in range(4):
        n = sum(gen_results_k0[gen])
        print(f"  {gen:<6} {n:<12} {n/len(k0_indices):.1%}")

    # Transition matrix
    print(f"\n  Transition Matrix (all 200 questions):")
    print(f"  {'Gen':<6} {'C->C':<8} {'C->W':<8} {'W->C':<8} {'W->W':<8}")
    print(f"  {'-'*40}")
    for gen in range(1, 4):
        prev = gen_results_all[gen - 1]
        curr = gen_results_all[gen]
        cc = sum(1 for p, c in zip(prev, curr) if p and c)
        cw = sum(1 for p, c in zip(prev, curr) if p and not c)
        wc = sum(1 for p, c in zip(prev, curr) if not p and c)
        ww = sum(1 for p, c in zip(prev, curr) if not p and not c)
        print(f"  {gen:<6} {cc:<8} {cw:<8} {wc:<8} {ww:<8}")

    # Save
    output = {
        "k0_size": len(k0_indices),
        "k0_indices": k0_indices,
        "retention_per_gen": {gen: sum(gen_results_k0[gen]) for gen in range(4)},
        "global_acc_per_gen": {gen: sum(gen_results_all[gen]) for gen in range(4)},
    }
    out_path = OUTPUT_DIR / "retention_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
