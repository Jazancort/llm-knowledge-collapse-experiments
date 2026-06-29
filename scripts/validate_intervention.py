"""Validation: C5 token-matched control + C3 multi-seed replication.

C5: Train on NORMAL synthetic data but downsampled to match C3's token budget.
    If C5 ~ C1: filtering quality matters. If C5 ~ C3: just token reduction.

C3 seeds 137/256: Replicate length-filtered intervention across seeds.

Run on Athena:
  uv run python scripts/validate_intervention.py
"""
import sys, gc, json, time, re, math
from pathlib import Path
from collections import Counter

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

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_SIZE = 2000
EVAL_SIZE = 200
GENERATIONS = 5
RANK = 256
LR = 1e-5

NUMBER_WORDS = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
                "six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}

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
            out = model.generate(**inputs, max_new_tokens=30, temperature=0.7, do_sample=True,
                                 top_p=0.9, pad_token_id=tokenizer.pad_token_id)
        for seq in out:
            synthetic.append(tokenizer.decode(seq, skip_special_tokens=False))
    return synthetic

def filter_by_length(synthetic_texts, max_words=5):
    """C3: keep only short responses."""
    filtered = []
    for text in synthetic_texts:
        matches = re.findall(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, re.DOTALL)
        if not matches:
            filtered.append(text)
            continue
        response = matches[-1].strip()
        if 1 <= len(response.split()) <= max_words:
            filtered.append(text)
    # Pad back to original size by duplicating
    if len(filtered) < len(synthetic_texts) * 0.5:
        while len(filtered) < len(synthetic_texts):
            filtered.append(filtered[np.random.randint(len(filtered))])
    return filtered[:len(synthetic_texts)]

def downsample_to_token_budget(synthetic_texts, target_tokens):
    """C5: randomly remove examples to match target token budget."""
    # Shuffle and take examples until we hit the budget
    indices = list(range(len(synthetic_texts)))
    np.random.shuffle(indices)
    selected = []
    total = 0
    for idx in indices:
        tokens = len(synthetic_texts[idx].split())
        if total + tokens > target_tokens:
            break
        selected.append(synthetic_texts[idx])
        total += tokens
    return selected if selected else synthetic_texts[:100]

def defrag():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    time.sleep(2)

def get_batch_size():
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb >= 20: return 4, 4
    elif total_gb >= 12: return 2, 8
    else: return 1, 16

def run_condition(cond_name, seed, process_fn, train_questions, k0_questions, k0_answers, output_dir):
    """Run one condition: train Gen1-5 with process_fn applied to synthetic data each gen."""
    result_path = output_dir / f"{cond_name}_seed{seed}.json"
    gen_results = json.load(open(result_path)) if result_path.exists() else []
    if len(gen_results) >= GENERATIONS:
        print(f"\n  [{cond_name} seed={seed}] Done ({len(gen_results)} gens).")
        return gen_results

    start_gen = len(gen_results) + 1
    print(f"\n{'='*60}")
    print(f"  {cond_name} seed={seed} — Gen{start_gen} to Gen{GENERATIONS}")
    print(f"{'='*60}")

    # Find last synthetic
    syn_path = output_dir / f"syn_{cond_name}_s{seed}_gen{start_gen-1}.json"
    if not syn_path.exists() and start_gen == 1:
        syn_path = output_dir / f"synthetic_gen0_seed{seed}.json"

    for gen in range(start_gen, GENERATIONS + 1):
        prev_synthetic = json.load(open(syn_path))
        t0 = time.time()
        print(f"\n  [Gen {gen}] ({len(prev_synthetic)} examples, {sum(len(t.split()) for t in prev_synthetic)} tokens)")

        # Load model
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                                      device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        torch.manual_seed(seed + gen)
        lora_config = LoraConfig(r=RANK, lora_alpha=RANK*2, lora_dropout=0.05,
                                 target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

        # Train
        def tok_fn(ex):
            return tok(ex["text"], truncation=True, max_length=256, padding="max_length")
        train_ds = Dataset.from_dict({"text": prev_synthetic})
        train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
        train_ds.set_format("torch")

        bs, accum = get_batch_size()
        trainer = Trainer(model=model, args=TrainingArguments(
            output_dir=str(output_dir / "tmp"), num_train_epochs=2,
            per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
            learning_rate=LR, bf16=True, logging_steps=9999,
            save_strategy="no", report_to="none", seed=seed + gen),
            train_dataset=train_ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
        trainer.train()
        model.eval()
        del trainer, train_ds; gc.collect(); torch.cuda.empty_cache()

        # Eval
        k0_res = evaluate_k0(model, tok, k0_questions, k0_answers)
        ret = sum(k0_res)
        print(f"    K0: {ret}/{len(k0_questions)} ({ret/len(k0_questions):.1%})")

        # Generate normal synthetic then apply process_fn
        raw_syn = generate_synthetic(model, tok, train_questions, seed_offset=seed + gen + 100)
        processed = process_fn(raw_syn)
        syn_path = output_dir / f"syn_{cond_name}_s{seed}_gen{gen}.json"
        json.dump(processed, open(syn_path, "w"))
        del model, raw_syn, processed; defrag()

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.0f}s")
        gen_results.append({"gen": gen, "retention": ret, "k0_results": k0_res, "time": elapsed})
        json.dump(gen_results, open(result_path, "w"), indent=2)

    return gen_results


def main():
    output_dir = Path(__file__).parent.parent / "outputs" / "intervention_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)
    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    k0_indices = json.load(open(Path(__file__).parent.parent / "outputs" / "fft_lr_sweep" / "k0_indices.json"))
    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]
    print(f"K0 size: {len(k0_indices)}")

    # C3's average token budget per gen was ~54,466
    C3_TOKEN_BUDGET = 54500

    # Define conditions
    def c5_process(synthetic):
        """Token-matched random downsample (no quality filter)."""
        return downsample_to_token_budget(synthetic, C3_TOKEN_BUDGET)

    def c3_process(synthetic):
        """Length filter (same as original C3)."""
        return filter_by_length(synthetic, max_words=5)

    # Generate gen0 synthetic per seed
    for seed in [15, 137, 256]:
        syn0 = output_dir / f"synthetic_gen0_seed{seed}.json"
        if not syn0.exists():
            print(f"Generating Gen0 for seed {seed}...")
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                                          device_map="auto", torch_dtype=torch.bfloat16)
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            model.eval()
            syn = generate_synthetic(model, tok, train_questions, seed_offset=seed)
            json.dump(syn, open(syn0, "w"))
            del model; defrag()

    # Run C5 token-matched (seed 15 only)
    run_condition("C5_token_matched", 15, c5_process, train_questions, k0_questions, k0_answers, output_dir)

    # Run C3 length-filtered (seeds 137, 256)
    for seed in [137, 256]:
        run_condition("C3_filtered", seed, c3_process, train_questions, k0_questions, k0_answers, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n  {'Condition':<25} {'Seed':<6} {'Gen1':>5} {'Gen2':>5} {'Gen3':>5} {'Gen4':>5} {'Gen5':>5}")
    print(f"  {'-'*60}")
    print(f"  {'C1_normal (known)':<25} {'15':<6} {'73':>5} {'72':>5} {'70':>5} {'68':>5} {'65':>5}")
    print(f"  {'C3_filtered (known)':<25} {'15':<6} {'74':>5} {'74':>5} {'71':>5} {'71':>5} {'73':>5}")

    for cond in ["C5_token_matched", "C3_filtered"]:
        for seed in [15, 137, 256]:
            rp = output_dir / f"{cond}_seed{seed}.json"
            if rp.exists():
                data = json.load(open(rp))
                rets = [str(d["retention"]) for d in data]
                print(f"  {cond:<25} {seed:<6} {'  '.join(rets):>30}")

    print(f"\n  INTERPRETATION:")
    print(f"    C5 ~ C1: token reduction alone does NOT explain C3's gain")
    print(f"    C5 ~ C3: gain was just from fewer tokens (bad news)")
    print(f"    C3 seeds consistent: intervention is robust")


if __name__ == "__main__":
    main()
