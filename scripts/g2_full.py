"""G2-Full: 3 generations with REAL data (independent shards).

Answers: "Does the model stabilize under ANY repeated fine-tuning,
or is the stabilization specific to synthetic recursion?"

Protocol:
- Gen 1: Train adapter on Shard 1 (real TriviaQA, samples 0-2000)
- Gen 2: Train NEW adapter on Shard 2 (real TriviaQA, samples 2000-4000)
- Gen 3: Train NEW adapter on Shard 3 (real TriviaQA, samples 4000-6000)

Same eval set (samples 6000-6200) and probe set (samples 6200-6300) across all.
Same LoRA config (r=16, q_proj+v_proj), same LR, same epochs.

If G2 keeps drifting while G1 stabilized → recursion causes anchoring
If G2 also stabilizes → QLoRA is inherently stabilizing regardless of data

Run: uv run python scripts/g2_full.py
"""

import sys
import gc
import json
import re
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
MONITOR_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 26]
SEED = 15
SHARD_SIZE = 2000
NUM_SHARDS = 3
EVAL_SIZE = 200
PROBE_SIZE = 100
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "g2_full"

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


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtX, YtY = X @ X.T, Y @ Y.T
    return float(np.trace(XtX @ YtY) / (np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro') + 1e-10))


class ProbeExtractor:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.hooks = []
        self.buffer = {l: {"global": [], "f1": []} for l in layers}
        self.end_indices = []

    def _hook(self, idx):
        def fn(module, inp, out):
            t = out[0].detach()
            for i in range(t.size(0)):
                end = min(self.end_indices[i], t.size(1) - 1)
                self.buffer[idx]["global"].append(t[i, :end+1, :].mean(0).cpu().float().numpy())
                self.buffer[idx]["f1"].append(t[i, end, :].cpu().float().numpy())
        return fn

    def register(self):
        m = self.model
        if hasattr(m, "base_model"):
            m = m.base_model
        for name, child in m.named_modules():
            if name.endswith("layers") and hasattr(child, "__getitem__"):
                for idx in self.layers:
                    self.hooks.append(child[idx].register_forward_hook(self._hook(idx)))
                return

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def reset(self):
        self.buffer = {l: {"global": [], "f1": []} for l in self.layers}

    def stacked(self, idx, key):
        return np.stack(self.buffer[idx][key])


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


def extract_reps(model, tokenizer, questions, layers):
    extractor = ProbeExtractor(model, layers)
    extractor.register()
    for i, q in enumerate(questions):
        prompt = format_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        extractor.end_indices = [inputs.input_ids.shape[1] - 1]
        with torch.no_grad():
            model(**inputs)
        if (i + 1) % 50 == 0:
            print(f"      Probe {i+1}/{len(questions)}...", flush=True)
    reps = {idx: {"global": extractor.stacked(idx, "global"),
                  "f1": extractor.stacked(idx, "f1")} for idx in layers}
    extractor.clear()
    return reps


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
            c = sum(results)
            print(f"      Eval {i+1}/{len(questions)} (acc: {c/(i+1):.1%})", flush=True)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=15)
    args = parser.parse_args()
    run_seed = args.seed

    print("=" * 60)
    print(f"G2-FULL: 3 GENERATIONS WITH INDEPENDENT REAL DATA SHARDS (seed={run_seed})")
    print("=" * 60)

    OUTPUT_DIR_SEED = Path(__file__).parent.parent / "outputs" / f"g2_full_seed{run_seed}"
    OUTPUT_DIR_SEED.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    # Load TriviaQA — FIXED shuffle (seed=15) so eval set is always the same
    print("\n[SETUP] Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)  # ALWAYS 15 for dataset split consistency

    # Shards: 0-2000, 2000-4000, 4000-6000
    # Eval: 6000-6200, Probe: 6200-6300
    total_train = SHARD_SIZE * NUM_SHARDS
    eval_start = total_train
    probe_start = eval_start + EVAL_SIZE

    eval_questions = [ds[i]["question"] for i in range(eval_start, eval_start + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(eval_start, eval_start + EVAL_SIZE)]
    probe_questions = [ds[i]["question"] for i in range(probe_start, probe_start + PROBE_SIZE)]

    print(f"  Shards: {NUM_SHARDS} x {SHARD_SIZE}")
    print(f"  Eval: {EVAL_SIZE}, Probe: {PROBE_SIZE}")

    # --- Gen 0 ---
    print(f"\n[Gen 0] Base model...")
    model, tokenizer = load_base()

    print("    Evaluating...")
    gen0_results = evaluate_questions(model, tokenizer, eval_questions, eval_answers)
    k0_indices = [i for i, r in enumerate(gen0_results) if r]
    print(f"    Accuracy: {sum(gen0_results)}/{EVAL_SIZE} = {sum(gen0_results)/EVAL_SIZE:.1%}")
    print(f"    K0 size: {len(k0_indices)}")

    print("    Extracting reps...")
    gen0_reps = extract_reps(model, tokenizer, probe_questions, MONITOR_LAYERS)

    del model
    gc.collect(); torch.cuda.empty_cache()

    # Track
    all_gen_results = {0: gen0_results}
    cka_vs_gen0 = {}
    retention = {0: len(k0_indices)}
    transitions = {}

    # --- Gen 1-3 with independent real shards ---
    for gen in range(1, NUM_SHARDS + 1):
        shard_start = (gen - 1) * SHARD_SIZE
        shard_end = gen * SHARD_SIZE
        print(f"\n[Gen {gen}] Training on REAL shard [{shard_start}:{shard_end}]...")

        shard_questions = [ds[i]["question"] for i in range(shard_start, shard_end)]
        shard_answers = [ds[i]["answer"]["value"] for i in range(shard_start, shard_end)]

        model, tokenizer = load_base()

        # Format real training data
        train_texts = []
        for q, a in zip(shard_questions, shard_answers):
            prompt = format_prompt(tokenizer, q)
            train_texts.append(prompt + a)

        # Train
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        peft_model = get_peft_model(model, lora_config)

        def tok_fn(ex):
            return tokenizer(ex["text"], truncation=True, max_length=256, padding="max_length")

        train_ds = Dataset.from_dict({"text": train_texts})
        train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
        train_ds.set_format("torch")

        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR_SEED / f"gen{gen}_tmp"),
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            bf16=True,
            logging_steps=100,
            save_strategy="no",
            report_to="none",
            seed=run_seed + gen,
        )
        trainer = Trainer(
            model=peft_model, args=args, train_dataset=train_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        peft_model.eval()

        # Evaluate
        print(f"    Evaluating...")
        gen_results = evaluate_questions(peft_model, tokenizer, eval_questions, eval_answers)
        all_gen_results[gen] = gen_results
        acc = sum(gen_results) / EVAL_SIZE
        print(f"    Accuracy: {sum(gen_results)}/{EVAL_SIZE} = {acc:.1%}")

        # K0 retention
        k0_correct = sum(gen_results[i] for i in k0_indices)
        retention[gen] = k0_correct
        print(f"    K0 retention: {k0_correct}/{len(k0_indices)} = {k0_correct/len(k0_indices):.1%}")

        # Transitions
        prev = all_gen_results[gen - 1]
        curr = gen_results
        cc = sum(1 for p, c in zip(prev, curr) if p and c)
        cw = sum(1 for p, c in zip(prev, curr) if p and not c)
        wc = sum(1 for p, c in zip(prev, curr) if not p and c)
        ww = sum(1 for p, c in zip(prev, curr) if not p and not c)
        transitions[gen] = {"C->C": cc, "C->W": cw, "W->C": wc, "W->W": ww}
        print(f"    Transitions: C->C={cc} C->W={cw} W->C={wc} W->W={ww}")

        # CKA
        print(f"    Extracting reps...")
        gen_reps = extract_reps(peft_model, tokenizer, probe_questions, MONITOR_LAYERS)
        cka_vs_gen0[gen] = {
            idx: {"global": linear_cka(gen0_reps[idx]["global"], gen_reps[idx]["global"]),
                  "f1": linear_cka(gen0_reps[idx]["f1"], gen_reps[idx]["f1"])}
            for idx in MONITOR_LAYERS
        }
        print(f"    CKA-Factual (layer 13): {cka_vs_gen0[gen][13]['f1']:.6f}")

        del peft_model, trainer, model
        gc.collect(); torch.cuda.empty_cache()

    # --- Report ---
    print("\n" + "=" * 60)
    print("G2-FULL FINAL REPORT")
    print("=" * 60)

    print(f"\n  K0 Retention (N={len(k0_indices)}):")
    for gen in range(NUM_SHARDS + 1):
        n = retention[gen]
        print(f"    Gen {gen}: {n}/{len(k0_indices)} = {n/len(k0_indices):.1%}")

    print(f"\n  Transitions:")
    for gen in range(1, NUM_SHARDS + 1):
        t = transitions[gen]
        print(f"    Gen {gen}: C->C={t['C->C']} C->W={t['C->W']} W->C={t['W->C']} W->W={t['W->W']}")

    print(f"\n  CKA-Factual vs Gen0 (layer 13):")
    for gen in range(1, NUM_SHARDS + 1):
        print(f"    Gen {gen}: {cka_vs_gen0[gen][13]['f1']:.6f}")

    print(f"\n  COMPARISON (Layer 13 CKA-Factual):")
    print(f"    G1 (synthetic, from M2): ~0.9836")
    for gen in range(1, NUM_SHARDS + 1):
        print(f"    G2 Gen {gen} (real shard {gen}): {cka_vs_gen0[gen][13]['f1']:.6f}")

    # Save
    results = {
        "k0_size": len(k0_indices),
        "retention": retention,
        "transitions": transitions,
        "cka_vs_gen0_layer13": {gen: cka_vs_gen0[gen][13] for gen in range(1, NUM_SHARDS + 1)},
    }
    with open(OUTPUT_DIR_SEED / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {OUTPUT_DIR_SEED / 'results.json'}")


if __name__ == "__main__":
    main()
