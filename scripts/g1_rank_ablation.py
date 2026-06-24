"""G1 Rank Ablation: Test if higher LoRA rank breaks the stabilization.

Hypothesis: Low rank (r=16) acts as implicit regularizer preventing collapse.
If r=64 causes degradation, rank is the protective mechanism.
If r=64 maintains homeostasis, stabilization is intrinsic to the base model.

Same protocol as g1_gen10 but with r=64. Runs 5 generations (enough to see signal).

Run: uv run python scripts/g1_rank_ablation.py [--rank 64] [--seed 15]
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
        if (i + 8) % 500 == 0:
            print(f"      Gen {min(i+8, len(questions))}/{len(questions)}...", flush=True)
    return synthetic


def compute_lora_spectrum(model):
    ranks, s_norms, f_norms = [], [], []
    for name, param in model.named_parameters():
        if "lora_A" in name and param.requires_grad:
            b_name = name.replace("lora_A", "lora_B")
            b_param = dict(model.named_parameters()).get(b_name)
            if b_param is not None:
                AB = b_param.data.float().cpu() @ param.data.float().cpu()
                svs = torch.linalg.svdvals(AB)
                svs_n = svs / (svs.sum() + 1e-10)
                ranks.append(torch.exp(-(svs_n * torch.log(svs_n + 1e-10)).sum()).item())
                s_norms.append(svs[0].item())
                f_norms.append(torch.norm(AB, 'fro').item())
    return {"eff_rank": np.mean(ranks), "spectral": np.mean(s_norms), "frobenius": np.mean(f_norms)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--generations", type=int, default=5)
    args = parser.parse_args()
    rank = args.rank
    seed = args.seed
    NUM_GENERATIONS = args.generations

    output_dir = Path(__file__).parent.parent / "outputs" / f"g1_rank{rank}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"G1 RANK ABLATION: r={rank}, seed={seed}, {NUM_GENERATIONS} generations")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data (same split as all other experiments)
    print("\n[SETUP] Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)

    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    # Check resume
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
        prev_synthetic = generate_synthetic(model, tokenizer, train_questions, seed_offset=seed)
        with open(output_dir / "synthetic_gen0.json", "w") as f:
            json.dump(prev_synthetic, f)

        results = [{"generation": 0, "accuracy": acc, "k0_size": len(k0_indices),
                    "k0_indices": k0_indices, "retention": len(k0_indices),
                    "transitions": None, "adapter": None}]
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        del model
        gc.collect(); torch.cuda.empty_cache()
        start_gen = 1

    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]

    # Gen 1-N
    for gen in range(start_gen, NUM_GENERATIONS + 1):
        print(f"\n[Gen {gen}] (r={rank})")
        model, tokenizer = load_base()

        print(f"    Training LoRA r={rank}...")
        torch.manual_seed(seed + gen)
        lora_config = LoraConfig(
            r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        peft_model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        print(f"    Trainable params: {trainable:,}")

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

        # Adapter health
        spectrum = compute_lora_spectrum(peft_model)
        print(f"    Effective rank: {spectrum['eff_rank']:.2f} / {rank}")

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

        # Transitions
        prev_all = results[-1].get("all_results")
        if prev_all:
            cc = sum(1 for p, c in zip(prev_all, all_res) if p and c)
            cw = sum(1 for p, c in zip(prev_all, all_res) if p and not c)
            wc = sum(1 for p, c in zip(prev_all, all_res) if not p and c)
            ww = sum(1 for p, c in zip(prev_all, all_res) if not p and not c)
            trans = {"C->C": cc, "C->W": cw, "W->C": wc, "W->W": ww}
            print(f"    Transitions: C->C={cc} C->W={cw} W->C={wc} W->W={ww}")
        else:
            trans = None

        # Generate next synthetic
        print("    Generating synthetic...")
        new_synthetic = generate_synthetic(peft_model, tokenizer, train_questions, seed_offset=seed + gen + 100)
        with open(output_dir / f"synthetic_gen{gen}.json", "w") as f:
            json.dump(new_synthetic, f)

        results.append({"generation": gen, "accuracy": acc, "retention": ret,
                        "transitions": trans, "adapter": spectrum, "all_results": all_res})
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        prev_synthetic = new_synthetic
        del peft_model, trainer, model
        gc.collect(); torch.cuda.empty_cache()

    # Final report
    print("\n" + "=" * 60)
    print(f"RANK ABLATION FINAL (r={rank})")
    print("=" * 60)
    print(f"\n  {'Gen':<6} {'Acc':<8} {'Retention':<12} {'Eff.Rank':<10} {'C->W':<6} {'W->C'}")
    print(f"  {'-'*48}")
    for r in results:
        t = r.get("transitions")
        cw = t["C->W"] if t else "-"
        wc = t["W->C"] if t else "-"
        er = f"{r['adapter']['eff_rank']:.1f}" if r.get("adapter") else "-"
        print(f"  {r['generation']:<6} {r['accuracy']:<8.1%} {r['retention']}/{len(k0_indices):<9} {er:<10} {cw:<6} {wc}")

    # Compare with r=16 baseline
    r16_path = Path(__file__).parent.parent / "outputs" / f"g1_gen10_seed{seed}" / "results.json"
    if r16_path.exists():
        with open(r16_path) as f:
            r16 = json.load(f)
        print(f"\n  COMPARISON (Gen 5, r=16 vs r={rank}):")
        r16_g5 = next((x for x in r16 if x["generation"] == 5), None)
        r_new_g5 = next((x for x in results if x["generation"] == 5), None)
        if r16_g5 and r_new_g5:
            print(f"    r=16:  retention={r16_g5['retention']}/{len(k0_indices)}, acc={r16_g5['accuracy']:.1%}")
            print(f"    r={rank}: retention={r_new_g5['retention']}/{len(k0_indices)}, acc={r_new_g5['accuracy']:.1%}")


if __name__ == "__main__":
    main()
