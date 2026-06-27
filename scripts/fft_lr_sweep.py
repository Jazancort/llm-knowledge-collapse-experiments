"""FFT LR Sweep: Find where FFT engages and compare drift to QLoRA.

Runs FFT at multiple LRs for 3 generations, tracking weight drift per gen.
Then compare to QLoRA r=16 drift to find the "matched" FFT LR.

Run on Athena:
  uv run python scripts/fft_lr_sweep.py
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
LRS = [1e-6, 5e-6, 1e-5, 2e-5]
GENERATIONS = 3
SEED = 15

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

def compute_weight_drift(model, initial_state):
    total_drift_sq = 0.0
    total_norm_sq = 0.0
    for name, param in model.named_parameters():
        if name in initial_state and param.requires_grad:
            diff = (param.data.float().cpu() - initial_state[name])
            total_drift_sq += diff.norm().item() ** 2
            total_norm_sq += initial_state[name].norm().item() ** 2
    return (total_drift_sq ** 0.5, (total_drift_sq / (total_norm_sq + 1e-10)) ** 0.5)

def compute_lora_drift(model):
    ranks = []
    total_norm = 0.0
    for name, param in model.named_parameters():
        if "lora_A" in name and param.requires_grad:
            b_name = name.replace("lora_A", "lora_B")
            b_param = dict(model.named_parameters()).get(b_name)
            if b_param is not None:
                AB = b_param.data.float().cpu() @ param.data.float().cpu()
                total_norm += AB.norm().item() ** 2
                svs = torch.linalg.svdvals(AB)
                svs_n = svs / (svs.sum() + 1e-10)
                ranks.append(torch.exp(-(svs_n * torch.log(svs_n + 1e-10)).sum()).item())
    return np.mean(ranks), total_norm ** 0.5


def main():
    output_dir = Path(__file__).parent.parent / "outputs" / "fft_lr_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)
    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    # Gen0 baseline + synthetic (shared across all LRs)
    syn_path = output_dir / "synthetic_gen0.json"
    if not syn_path.exists():
        print("[Gen 0] Generating baseline...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model.eval()
        gen0_res = evaluate_k0(model, tok, eval_questions, eval_answers)
        k0_indices = [i for i, r in enumerate(gen0_res) if r]
        acc = sum(gen0_res) / len(gen0_res)
        print(f"  Baseline: {acc:.1%}, K0={len(k0_indices)}")
        synthetic = generate_synthetic(model, tok, train_questions, seed_offset=SEED)
        json.dump(synthetic, open(syn_path, "w"))
        json.dump(k0_indices, open(output_dir / "k0_indices.json", "w"))
        del model; gc.collect(); torch.cuda.empty_cache()
    else:
        k0_indices = json.load(open(output_dir / "k0_indices.json"))
        print(f"[Gen 0] Loaded (K0={len(k0_indices)})")

    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]

    all_results = {}

    # === FFT sweep ===
    for lr in LRS:
        lr_key = f"fft_lr{lr:.0e}"
        result_path = output_dir / f"{lr_key}.json"
        if result_path.exists():
            all_results[lr_key] = json.load(open(result_path))
            print(f"\n[{lr_key}] Already done, skipping.")
            continue

        print(f"\n{'='*50}")
        print(f"FFT LR={lr:.0e}")
        print(f"{'='*50}")

        prev_synthetic = json.load(open(syn_path))
        gen_results = []

        for gen in range(1, GENERATIONS + 1):
            t0 = time.time()
            print(f"\n  [Gen {gen}]")
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            tok.padding_side = "left"

            initial_state = {n: p.data.float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    Params: {trainable:,}")

            def tok_fn(ex):
                return tok(ex["text"], truncation=True, max_length=256, padding="max_length")
            train_ds = Dataset.from_dict({"text": prev_synthetic})
            train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
            train_ds.set_format("torch")

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=str(output_dir / "tmp"), num_train_epochs=2,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8,
                    learning_rate=lr, bf16=True, logging_steps=9999,
                    save_strategy="no", report_to="none", seed=SEED + gen,
                    gradient_checkpointing=True,
                    optim="paged_adamw_8bit",
                ),
                train_dataset=train_ds,
                data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
            )
            trainer.train()
            model.eval()

            abs_drift, rel_drift = compute_weight_drift(model, initial_state)
            print(f"    Drift: abs={abs_drift:.4f} rel={rel_drift:.6f}")
            del initial_state

            k0_res = evaluate_k0(model, tok, k0_questions, k0_answers)
            ret = sum(k0_res)
            print(f"    K0: {ret}/{len(k0_indices)} ({ret/len(k0_indices):.1%})")

            prev_synthetic = generate_synthetic(model, tok, train_questions, seed_offset=SEED + gen + 100)
            elapsed = time.time() - t0
            print(f"    Time: {elapsed:.0f}s")

            gen_results.append({"gen": gen, "retention": ret, "abs_drift": abs_drift,
                                "rel_drift": rel_drift, "time": elapsed})
            del model, trainer; gc.collect(); torch.cuda.empty_cache()

        all_results[lr_key] = gen_results
        json.dump(gen_results, open(result_path, "w"), indent=2)

    # === QLoRA r=16 baseline (for drift comparison) ===
    qlora_path = output_dir / "qlora_r16.json"
    if not qlora_path.exists():
        print(f"\n{'='*50}")
        print("QLoRA r=16 (drift baseline)")
        print(f"{'='*50}")

        prev_synthetic = json.load(open(syn_path))
        gen_results = []

        for gen in range(1, GENERATIONS + 1):
            t0 = time.time()
            print(f"\n  [Gen {gen}]")
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            tok.padding_side = "left"

            torch.manual_seed(SEED + gen)
            lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
            model.enable_input_require_grads()
            model = get_peft_model(model, lora_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    Params: {trainable:,}")

            def tok_fn(ex):
                return tok(ex["text"], truncation=True, max_length=256, padding="max_length")
            train_ds = Dataset.from_dict({"text": prev_synthetic})
            train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
            train_ds.set_format("torch")

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=str(output_dir / "tmp"), num_train_epochs=2,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8,
                    learning_rate=1e-5, bf16=True, logging_steps=9999,
                    save_strategy="no", report_to="none", seed=SEED + gen,
                ),
                train_dataset=train_ds,
                data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
            )
            trainer.train()
            model.eval()

            eff_rank, lora_norm = compute_lora_drift(model)
            print(f"    LoRA norm: {lora_norm:.4f}, eff_rank: {eff_rank:.2f}")

            k0_res = evaluate_k0(model, tok, k0_questions, k0_answers)
            ret = sum(k0_res)
            print(f"    K0: {ret}/{len(k0_indices)} ({ret/len(k0_indices):.1%})")

            prev_synthetic = generate_synthetic(model, tok, train_questions, seed_offset=SEED + gen + 100)
            elapsed = time.time() - t0
            print(f"    Time: {elapsed:.0f}s")

            gen_results.append({"gen": gen, "retention": ret, "eff_rank": eff_rank,
                                "lora_norm": lora_norm, "time": elapsed})
            del model, trainer; gc.collect(); torch.cuda.empty_cache()

        all_results["qlora_r16"] = gen_results
        json.dump(gen_results, open(qlora_path, "w"), indent=2)
    else:
        all_results["qlora_r16"] = json.load(open(qlora_path))
        print("\n[QLoRA r=16] Already done, skipping.")

    # === Summary ===
    print("\n" + "=" * 70)
    print("FFT LR SWEEP SUMMARY")
    print("=" * 70)
    print(f"\n  {'Config':<20} {'Gen3 Ret':<12} {'Drift (rel)':<14} {'Loss trajectory'}")
    print(f"  {'-'*55}")
    for key, gens in sorted(all_results.items()):
        last = gens[-1]
        k0 = len(k0_indices)
        ret = last["retention"]
        drift = last.get("rel_drift", last.get("lora_norm", 0))
        print(f"  {key:<20} {ret}/{k0} ({ret/k0*100:.1f}%)  drift={drift:.6f}")

    print("\n  DECISION: Choose FFT LR whose drift is closest to QLoRA lora_norm.")
    print("  Then run that LR + one higher at n=3, Gen5.")


if __name__ == "__main__":
    main()
