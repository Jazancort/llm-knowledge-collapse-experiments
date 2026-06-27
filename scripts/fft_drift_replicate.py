"""Replicate drift-matched pair: QLoRA r=16 vs FFT LR=1e-6, seeds 137+256, Gen5.

This is the final validation: does the ~4pp structural benefit hold across seeds?

Run on Athena:
  uv run python scripts/fft_drift_replicate.py
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
SEEDS = [137, 256]
GENERATIONS = 5
FFT_LR = 1e-6
QLORA_LR = 1e-5

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

def evaluate_k0(model, tokenizer, k0_questions, k0_answers):
    results = []
    for q, gts in zip(k0_questions, k0_answers):
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
    for i in range(0, len(questions), 4):
        batch = questions[i:i+4]
        prompts = [format_prompt(tokenizer, q) for q in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, temperature=0.7, do_sample=True, top_p=0.9)
        for seq in out:
            synthetic.append(tokenizer.decode(seq, skip_special_tokens=False))
    return synthetic

def compute_weight_drift(model, initial_state):
    total_drift_sq = 0.0
    total_norm_sq = 0.0
    for name, param in model.named_parameters():
        if name in initial_state and param.requires_grad:
            diff = (param.data.float().cpu() - initial_state[name])
            total_drift_sq += diff.norm().item() ** 2
            total_norm_sq += initial_state[name].norm().item() ** 2
    return total_drift_sq ** 0.5, (total_drift_sq / total_norm_sq) ** 0.5 if total_norm_sq > 0 else 0.0

def compute_lora_norm(model):
    norms = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            norms.append(param.data.float().cpu().norm().item() ** 2)
    return sum(norms) ** 0.5

def run_fft(seed, generations, train_questions, k0_questions, k0_answers, output_dir):
    key = f"fft_lr1e-06_seed{seed}"
    result_path = output_dir / f"{key}.json"
    if result_path.exists():
        data = json.load(open(result_path))
        if len(data) >= generations:
            print(f"\n[{key}] Already done, skipping.")
            return data

    print(f"\n{'='*50}")
    print(f"FFT LR=1e-6, seed={seed}")
    print(f"{'='*50}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Gen0 synthetic for this seed
    syn_path = output_dir / f"synthetic_gen0_seed{seed}.json"
    if not syn_path.exists():
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model.eval()
        synthetic = generate_synthetic(model, tok, train_questions, seed_offset=seed)
        json.dump(synthetic, open(syn_path, "w"))
        del model; gc.collect(); torch.cuda.empty_cache()

    # Load existing progress
    gen_results = json.load(open(result_path)) if result_path.exists() else []
    start_gen = len(gen_results) + 1

    for gen in range(start_gen, generations + 1):
        # Load synthetic from previous gen
        prev_syn_path = output_dir / f"syn_{key}_gen{gen-1}.json" if gen > 1 else syn_path
        if not prev_syn_path.exists() and gen == 1:
            prev_syn_path = syn_path
        prev_synthetic = json.load(open(prev_syn_path))

        t0 = time.time()
        print(f"\n  [Gen {gen}]")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        initial_state = {n: p.data.float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

        def tok_fn(ex):
            return tok(ex["text"], truncation=True, max_length=256, padding="max_length")
        train_ds = Dataset.from_dict({"text": prev_synthetic})
        train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
        train_ds.set_format("torch")

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(output_dir / "tmp"), num_train_epochs=2,
                per_device_train_batch_size=4, gradient_accumulation_steps=4,
                learning_rate=FFT_LR, bf16=True, logging_steps=9999,
                save_strategy="no", report_to="none", seed=seed + gen,
                gradient_checkpointing=True, optim="paged_adamw_8bit",
            ),
            train_dataset=train_ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        )
        trainer.train()
        model.eval()

        abs_drift, rel_drift = compute_weight_drift(model, initial_state)
        del initial_state, trainer, train_ds
        gc.collect(); torch.cuda.empty_cache()
        print(f"    Drift: abs={abs_drift:.4f} rel={rel_drift:.6f}")

        k0_res = evaluate_k0(model, tok, k0_questions, k0_answers)
        ret = sum(k0_res)
        print(f"    K0: {ret}/{len(k0_questions)} ({ret/len(k0_questions):.1%})")

        # Generate synthetic and save to disk, then free model
        next_syn = generate_synthetic(model, tok, train_questions, seed_offset=seed + gen + 100)
        json.dump(next_syn, open(output_dir / f"syn_{key}_gen{gen}.json", "w"))
        del model, next_syn; gc.collect(); torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.0f}s")

        gen_results.append({"gen": gen, "retention": ret, "abs_drift": abs_drift, "rel_drift": rel_drift, "time": elapsed})
        # Save after every generation
        json.dump(gen_results, open(result_path, "w"), indent=2)

    return gen_results

def run_qlora(seed, generations, train_questions, k0_questions, k0_answers, output_dir):
    key = f"qlora_r16_seed{seed}"
    result_path = output_dir / f"{key}.json"
    if result_path.exists():
        data = json.load(open(result_path))
        if len(data) >= generations:
            print(f"\n[{key}] Already done, skipping.")
            return data

    print(f"\n{'='*50}")
    print(f"QLoRA r=16, seed={seed}")
    print(f"{'='*50}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    syn_path = output_dir / f"synthetic_gen0_seed{seed}.json"
    if not syn_path.exists():
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model.eval()
        synthetic = generate_synthetic(model, tok, train_questions, seed_offset=seed)
        json.dump(synthetic, open(syn_path, "w"))
        del model; gc.collect(); torch.cuda.empty_cache()

    # Load existing progress
    gen_results = json.load(open(result_path)) if result_path.exists() else []
    start_gen = len(gen_results) + 1

    for gen in range(start_gen, generations + 1):
        # Load synthetic from previous gen
        prev_syn_path = output_dir / f"syn_{key}_gen{gen-1}.json" if gen > 1 else syn_path
        if not prev_syn_path.exists() and gen == 1:
            prev_syn_path = syn_path
        prev_synthetic = json.load(open(prev_syn_path))

        t0 = time.time()
        print(f"\n  [Gen {gen}]")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        torch.manual_seed(seed + gen)
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

        def tok_fn(ex):
            return tok(ex["text"], truncation=True, max_length=256, padding="max_length")
        train_ds = Dataset.from_dict({"text": prev_synthetic})
        train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
        train_ds.set_format("torch")

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(output_dir / "tmp"), num_train_epochs=2,
                per_device_train_batch_size=4, gradient_accumulation_steps=4,
                learning_rate=QLORA_LR, bf16=True, logging_steps=9999,
                save_strategy="no", report_to="none", seed=seed + gen,
            ),
            train_dataset=train_ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        )
        trainer.train()
        model.eval()

        lora_norm = compute_lora_norm(model)
        del trainer, train_ds
        gc.collect(); torch.cuda.empty_cache()
        print(f"    LoRA norm: {lora_norm:.4f}")

        k0_res = evaluate_k0(model, tok, k0_questions, k0_answers)
        ret = sum(k0_res)
        print(f"    K0: {ret}/{len(k0_questions)} ({ret/len(k0_questions):.1%})")

        # Generate synthetic and save to disk
        next_syn = generate_synthetic(model, tok, train_questions, seed_offset=seed + gen + 100)
        json.dump(next_syn, open(output_dir / f"syn_{key}_gen{gen}.json", "w"))
        del model, next_syn; gc.collect(); torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.0f}s")

        gen_results.append({"gen": gen, "retention": ret, "lora_norm": lora_norm, "time": elapsed})
        # Save after every generation
        json.dump(gen_results, open(result_path, "w"), indent=2)

    return gen_results
    return gen_results


def main():
    output_dir = Path(__file__).parent.parent / "outputs" / "fft_drift_replicate"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)
    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    # Use same K0 as sweep
    sweep_k0_path = Path(__file__).parent.parent / "outputs" / "fft_lr_sweep" / "k0_indices.json"
    if sweep_k0_path.exists():
        k0_indices = json.load(open(sweep_k0_path))
    else:
        # Compute fresh
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model.eval()
        gen0_res = evaluate_k0(model, tok, eval_questions, eval_answers)
        k0_indices = [i for i, r in enumerate(gen0_res) if r]
        json.dump(k0_indices, open(output_dir / "k0_indices.json", "w"))
        del model; gc.collect(); torch.cuda.empty_cache()

    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]
    print(f"K0 size: {len(k0_indices)}")

    all_results = {}

    for seed in SEEDS:
        all_results[f"fft_seed{seed}"] = run_fft(seed, GENERATIONS, train_questions, k0_questions, k0_answers, output_dir)
        all_results[f"qlora_seed{seed}"] = run_qlora(seed, GENERATIONS, train_questions, k0_questions, k0_answers, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("DRIFT-MATCHED REPLICATION SUMMARY")
    print("=" * 70)
    print(f"\n  {'Config':<25} {'Gen3 Ret':<12} {'Gen5 Ret':<12} {'Perturbation'}")
    print(f"  {'-'*65}")

    # Include seed 15 from sweep for comparison
    sweep_dir = Path(__file__).parent.parent / "outputs" / "fft_lr_sweep"
    if (sweep_dir / "fft_lr1e-06.json").exists():
        s15_fft = json.load(open(sweep_dir / "fft_lr1e-06.json"))
        r3 = s15_fft[2]["retention"]
        d = s15_fft[2]["abs_drift"]
        print(f"  {'FFT 1e-6 seed=15':<25} {r3}/{len(k0_indices)} ({r3/len(k0_indices)*100:.1f}%)  {'—':<12} drift={d:.4f}")
    if (sweep_dir / "qlora_r16.json").exists():
        s15_q = json.load(open(sweep_dir / "qlora_r16.json"))
        r3 = s15_q[2]["retention"]
        n = s15_q[2]["lora_norm"]
        print(f"  {'QLoRA r=16 seed=15':<25} {r3}/{len(k0_indices)} ({r3/len(k0_indices)*100:.1f}%)  {'—':<12} norm={n:.4f}")

    for seed in SEEDS:
        fft_key = f"fft_seed{seed}"
        qlora_key = f"qlora_seed{seed}"
        if fft_key in all_results:
            gens = all_results[fft_key]
            r3 = gens[2]["retention"] if len(gens) >= 3 else "—"
            r5 = gens[4]["retention"] if len(gens) >= 5 else "—"
            d = gens[-1]["abs_drift"]
            print(f"  {f'FFT 1e-6 seed={seed}':<25} {r3}/{len(k0_indices)}          {r5}/{len(k0_indices)}          drift={d:.4f}")
        if qlora_key in all_results:
            gens = all_results[qlora_key]
            r3 = gens[2]["retention"] if len(gens) >= 3 else "—"
            r5 = gens[4]["retention"] if len(gens) >= 5 else "—"
            n = gens[-1]["lora_norm"]
            print(f"  {f'QLoRA r=16 seed={seed}':<25} {r3}/{len(k0_indices)}          {r5}/{len(k0_indices)}          norm={n:.4f}")

    print(f"\n  K0={len(k0_indices)}")
    print("\n  DECISION: If QLoRA consistently > FFT across 3 seeds → structural benefit confirmed.")
    print("            If gap disappears → magnitude is the full story.")


if __name__ == "__main__":
    main()
