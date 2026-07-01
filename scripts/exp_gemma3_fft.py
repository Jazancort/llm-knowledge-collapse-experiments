"""FFT learning rate sweep on Gemma 3 1B.

Tests whether perturbation-magnitude axis generalizes to Gemma 3.
3 LRs: 1e-6, 5e-6, 1e-5. Seed 15, 5 generations.

Run: uv run python scripts/exp_gemma3_fft.py
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
from datasets import load_dataset, Dataset

MODEL_NAME = "google/gemma-3-1b-it"
TRAIN_SIZE = 2000
EVAL_SIZE = 200
SEED = 15
GENERATIONS = 5
LRS = [1e-6, 5e-6, 1e-5]

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


def load_base_fft():
    """Load in bf16 (no quantization) for full fine-tuning."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def load_base_4bit():
    """Load in 4-bit for generation/eval only (saves VRAM)."""
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
    msgs = [{"role": "user", "content": f"Answer the following question in 5 words or less.\n{q}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def evaluate_questions(model, tokenizer, questions, answers):
    results = []
    for q, gts in zip(questions, answers):
        prompt = format_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append(exact_match(text, gts))
    return results


def generate_synthetic(model, tokenizer, questions, seed_offset=0):
    torch.manual_seed(seed_offset)
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
    return synthetic


def run_fft_config(lr, train_questions, k0_indices, k0_questions, k0_answers, output_dir):
    key = f"lr{lr:.0e}"
    print(f"\n{'='*60}")
    print(f"  GEMMA 3 FFT — LR={lr}, seed={SEED}")
    print(f"{'='*60}")

    result_path = output_dir / f"gemma3_fft_{key}.json"

    if result_path.exists():
        existing = json.loads(result_path.read_text())
        gens_done = [g["gen"] for g in existing.get("generations", [])]
        if GENERATIONS in gens_done:
            print(f"  Already complete. Skipping.")
            return existing

    # Gen 0: generate synthetic with 4-bit model
    model, tokenizer = load_base_4bit()
    prev_synthetic = generate_synthetic(model, tokenizer, train_questions, seed_offset=SEED)
    del model; defrag()

    gen_results = []
    for gen in range(1, GENERATIONS + 1):
        t0 = time.time()
        print(f"  [Gen {gen}] FFT LR={lr}")

        # Train with bf16 full model + gradient checkpointing
        model, tokenizer = load_base_fft()
        model.gradient_checkpointing_enable()
        model.train()

        def tok_fn(ex):
            return tokenizer(ex["text"], truncation=True, max_length=256, padding="max_length")
        train_ds = Dataset.from_dict({"text": prev_synthetic})
        train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
        train_ds.set_format("torch")

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(output_dir / "tmp"), num_train_epochs=2,
                per_device_train_batch_size=1, gradient_accumulation_steps=16,
                learning_rate=lr, bf16=True, logging_steps=9999,
                save_strategy="no", report_to="none", seed=SEED + gen,
                gradient_checkpointing=True,
                optim="paged_adamw_8bit",
                max_grad_norm=1.0,
            ),
            train_dataset=train_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        model.eval()

        k0_res = evaluate_questions(model, tokenizer, k0_questions, k0_answers)
        ret = sum(k0_res)
        elapsed = time.time() - t0
        print(f"    K0: {ret}/{len(k0_indices)} ({ret/len(k0_indices):.1%}) [{elapsed:.0f}s]")

        prev_synthetic = generate_synthetic(model, tokenizer, train_questions, seed_offset=SEED + gen + 100)

        gen_results.append({"gen": gen, "retention": ret, "time_sec": round(elapsed, 1)})

        result_data = {"model": MODEL_NAME, "method": "FFT", "lr": lr, "seed": SEED,
                       "k0_size": len(k0_indices), "generations": gen_results}
        result_path.write_text(json.dumps(result_data, indent=2))

        del trainer, model; defrag()

    return result_data


def main():
    output_dir = Path("outputs/gemma3_fft")
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)
    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    print("Establishing K0...")
    model, tokenizer = load_base_4bit()
    gen0_results = evaluate_questions(model, tokenizer, eval_questions, eval_answers)
    k0_indices = [i for i, r in enumerate(gen0_results) if r]
    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]
    print(f"K0 size: {len(k0_indices)} / {EVAL_SIZE}")
    del model; defrag()

    all_results = {}
    for lr in LRS:
        result = run_fft_config(lr, train_questions, k0_indices, k0_questions, k0_answers, output_dir)
        all_results[lr] = result

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Gemma 3 FFT LR Sweep")
    print("=" * 60)
    k0 = len(k0_indices)
    for lr in LRS:
        gens = all_results[lr]["generations"]
        g5 = gens[-1]["retention"]
        print(f"  LR={lr:.0e}: Gen5 = {g5}/{k0} ({g5/k0*100:.1f}%)")


if __name__ == "__main__":
    main()
