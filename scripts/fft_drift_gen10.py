"""Extend drift-matched pair to Gen10: FFT LR=1e-6 vs QLoRA r=16, 3 seeds.

Resumes from existing Gen5 data in fft_drift_replicate/.
Only runs Gen6-10 for each config that already has Gen1-5.

Run on Athena:
  uv run python scripts/fft_drift_gen10.py
"""
import sys, gc, json, time, shutil
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
import re

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_SIZE = 2000
EVAL_SIZE = 200
SEEDS = [15, 137, 256]
GENERATIONS = 10
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
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append(exact_match(text, gts))
    return results

def generate_synthetic(model, tokenizer, questions, seed_offset=0):
    torch.manual_seed(seed_offset)
    synthetic = []
    for i in tqdm(range(0, len(questions), 16), desc="    Synthetic", leave=False):
        batch = questions[i:i+16]
        prompts = [format_prompt(tokenizer, q) for q in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=30, temperature=0.7, do_sample=True, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
        for seq in out:
            synthetic.append(tokenizer.decode(seq, skip_special_tokens=False))
    return synthetic

def get_train_batch_size():
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb >= 20: return 8, 2
    elif total_gb >= 12: return 4, 4
    else: return 2, 8

def defrag():
    """Aggressive memory cleanup between generations to prevent OOM from fragmentation."""
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    # Force glibc to return memory to OS (Linux only)
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    time.sleep(2)

def compute_weight_drift(model, initial_state_path):
    state = torch.load(initial_state_path, map_location='cpu', weights_only=True)
    total_drift_sq = 0.0
    total_norm_sq = 0.0
    for name, param in model.named_parameters():
        if name in state and param.requires_grad:
            init_param = state[name].float()
            curr_param = param.data.float().cpu()
            diff = curr_param - init_param
            total_drift_sq += diff.norm().item() ** 2
            total_norm_sq += init_param.norm().item() ** 2
            del init_param, curr_param, diff
    del state; gc.collect()
    return total_drift_sq ** 0.5, (total_drift_sq / total_norm_sq) ** 0.5 if total_norm_sq > 0 else 0.0

def run_fft_gen10(seed, train_questions, k0_questions, k0_answers, output_dir, base_dir):
    key = f"fft_lr1e-06_seed{seed}"
    result_path = output_dir / f"{key}.json"

    # Load existing progress (from Gen5 run or partial Gen10)
    gen_results = []
    base_result = base_dir / f"{key}.json"
    if result_path.exists():
        gen_results = json.load(open(result_path))
    elif base_result.exists():
        gen_results = json.load(open(base_result))
        json.dump(gen_results, open(result_path, "w"), indent=2)

    if len(gen_results) >= GENERATIONS:
        print(f"\n[{key}] Already at Gen{len(gen_results)}, done.")
        return gen_results

    start_gen = len(gen_results) + 1
    print(f"\n{'='*50}")
    print(f"FFT LR=1e-6, seed={seed} — Gen{start_gen} to Gen{GENERATIONS}")
    print(f"{'='*50}")

    # Find last synthetic
    syn_path = None
    for g in range(start_gen - 1, 0, -1):
        candidate = output_dir / f"syn_{key}_gen{g}.json"
        if not candidate.exists():
            candidate = base_dir / f"syn_{key}_gen{g}.json"
        if candidate.exists():
            syn_path = candidate
            break
    if syn_path is None:
        syn_path = base_dir / f"synthetic_gen0_seed{seed}.json"
        if not syn_path.exists():
            syn_path = Path(__file__).parent.parent / "outputs" / "fft_lr_sweep" / "synthetic_gen0.json"

    for gen in range(start_gen, GENERATIONS + 1):
        prev_synthetic = json.load(open(syn_path))
        t0 = time.time()
        print(f"\n  [Gen {gen}]")

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        init_path = output_dir / "tmp_init.pt"
        torch.save({n: p.data.float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}, init_path)
        gc.collect()

        def tok_fn(ex):
            return tok(ex["text"], truncation=True, max_length=256, padding="max_length")
        train_ds = Dataset.from_dict({"text": prev_synthetic})
        train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
        train_ds.set_format("torch")

        bs, accum = get_train_batch_size()
        trainer = Trainer(model=model, args=TrainingArguments(
            output_dir=str(output_dir / "tmp"), num_train_epochs=2,
            per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
            learning_rate=FFT_LR, bf16=True, logging_steps=9999,
            save_strategy="no", report_to="none", seed=seed + gen,
            gradient_checkpointing=True, optim="paged_adamw_8bit"),
            train_dataset=train_ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
        trainer.train()
        model.eval()

        abs_drift, rel_drift = compute_weight_drift(model, init_path)
        del trainer, train_ds; gc.collect(); torch.cuda.empty_cache()
        if init_path.exists(): init_path.unlink()
        print(f"    Drift: abs={abs_drift:.4f}")

        k0_res = evaluate_k0(model, tok, k0_questions, k0_answers)
        ret = sum(k0_res)
        print(f"    K0: {ret}/{len(k0_questions)} ({ret/len(k0_questions):.1%})")

        # Save model, reload 4-bit for generation
        tmp_model = output_dir / "tmp_fft_model"
        model.save_pretrained(str(tmp_model))
        del model; gc.collect(); torch.cuda.empty_cache()

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        gen_model = AutoModelForCausalLM.from_pretrained(str(tmp_model), quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
        gen_model.eval()
        next_syn = generate_synthetic(gen_model, tok, train_questions, seed_offset=seed + gen + 100)
        syn_path = output_dir / f"syn_{key}_gen{gen}.json"
        json.dump(next_syn, open(syn_path, "w"))
        del gen_model, next_syn; gc.collect(); torch.cuda.empty_cache()
        if tmp_model.exists(): shutil.rmtree(tmp_model)

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.0f}s")
        gen_results.append({"gen": gen, "retention": ret, "abs_drift": abs_drift, "rel_drift": rel_drift, "time": elapsed, "k0_results": k0_res})
        json.dump(gen_results, open(result_path, "w"), indent=2)
        defrag()

    return gen_results

def run_qlora_gen10(seed, train_questions, k0_questions, k0_answers, output_dir, base_dir):
    key = f"qlora_r16_seed{seed}"
    result_path = output_dir / f"{key}.json"

    gen_results = []
    base_result = base_dir / f"{key}.json"
    if result_path.exists():
        gen_results = json.load(open(result_path))
    elif base_result.exists():
        gen_results = json.load(open(base_result))
        json.dump(gen_results, open(result_path, "w"), indent=2)

    if len(gen_results) >= GENERATIONS:
        print(f"\n[{key}] Already at Gen{len(gen_results)}, done.")
        return gen_results

    start_gen = len(gen_results) + 1
    print(f"\n{'='*50}")
    print(f"QLoRA r=16, seed={seed} — Gen{start_gen} to Gen{GENERATIONS}")
    print(f"{'='*50}")

    syn_path = None
    for g in range(start_gen - 1, 0, -1):
        candidate = output_dir / f"syn_{key}_gen{g}.json"
        if not candidate.exists():
            candidate = base_dir / f"syn_{key}_gen{g}.json"
        if candidate.exists():
            syn_path = candidate
            break
    if syn_path is None:
        syn_path = base_dir / f"synthetic_gen0_seed{seed}.json"
        if not syn_path.exists():
            syn_path = Path(__file__).parent.parent / "outputs" / "fft_lr_sweep" / "synthetic_gen0.json"

    for gen in range(start_gen, GENERATIONS + 1):
        prev_synthetic = json.load(open(syn_path))
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

        bs, accum = get_train_batch_size()
        trainer = Trainer(model=model, args=TrainingArguments(
            output_dir=str(output_dir / "tmp"), num_train_epochs=2,
            per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
            learning_rate=QLORA_LR, bf16=True, logging_steps=9999,
            save_strategy="no", report_to="none", seed=seed + gen),
            train_dataset=train_ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
        trainer.train()
        model.eval()

        # Compute lora B@A norm (correct metric)
        ba_norm_sq = 0.0
        params = dict(model.named_parameters())
        for name, param in model.named_parameters():
            if "lora_A" in name and param.requires_grad:
                b_name = name.replace("lora_A", "lora_B")
                if b_name in params:
                    BA = params[b_name].data.float().cpu() @ param.data.float().cpu()
                    ba_norm_sq += BA.norm().item() ** 2
        lora_ba_norm = ba_norm_sq ** 0.5

        del trainer, train_ds; gc.collect(); torch.cuda.empty_cache()
        print(f"    LoRA ||B@A||: {lora_ba_norm:.4f}")

        k0_res = evaluate_k0(model, tok, k0_questions, k0_answers)
        ret = sum(k0_res)
        print(f"    K0: {ret}/{len(k0_questions)} ({ret/len(k0_questions):.1%})")

        next_syn = generate_synthetic(model, tok, train_questions, seed_offset=seed + gen + 100)
        syn_path = output_dir / f"syn_{key}_gen{gen}.json"
        json.dump(next_syn, open(syn_path, "w"))
        del model, next_syn; gc.collect(); torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.0f}s")
        gen_results.append({"gen": gen, "retention": ret, "lora_ba_norm": lora_ba_norm, "time": elapsed, "k0_results": k0_res})
        json.dump(gen_results, open(result_path, "w"), indent=2)
        defrag()

    return gen_results


def main():
    output_dir = Path(__file__).parent.parent / "outputs" / "fft_drift_gen10"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(__file__).parent.parent / "outputs" / "fft_drift_replicate"

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)
    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    k0_indices = json.load(open(Path(__file__).parent.parent / "outputs" / "fft_lr_sweep" / "k0_indices.json"))
    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]
    print(f"K0 size: {len(k0_indices)}")

    for seed in SEEDS:
        run_fft_gen10(seed, train_questions, k0_questions, k0_answers, output_dir, base_dir)
        run_qlora_gen10(seed, train_questions, k0_questions, k0_answers, output_dir, base_dir)

    # Summary
    print("\n" + "=" * 70)
    print("GEN10 EXTENSION SUMMARY")
    print("=" * 70)
    for seed in SEEDS:
        for method in ["fft_lr1e-06", "qlora_r16"]:
            rp = output_dir / f"{method}_seed{seed}.json"
            if rp.exists():
                data = json.load(open(rp))
                rets = [d["retention"] for d in data]
                print(f"  {method} seed={seed}: {' → '.join(str(r) for r in rets)}")


if __name__ == "__main__":
    main()
