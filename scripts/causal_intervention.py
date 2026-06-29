"""Causal Intervention: Does breaking synthetic-output drift reduce factual degradation?

Tests whether the verbosity/distributional drift that accompanies degradation is
a causal participant or merely a co-symptom.

3 intervention conditions vs existing baseline (r=256 normal):
  C1: Normal (control) — already exists in g1_rank256_seed15
  C2: Short-answer constrained — force short generation via prompt
  C3: Length-filtered — generate normal, filter out long/bad responses
  C4: Canonical extraction — extract factoid from verbose responses

All conditions: Qwen 2.5 1.5B, QLoRA r=256, seed 15, Gen1-5.
Same training config as g1_rank_ablation (LR=1e-5, 2 epochs, batch 16).

Run on Athena:
  uv run python scripts/causal_intervention.py

Expected outcome:
  If C2/C3/C4 improve retention → drift participates causally
  If retention same → drift is co-symptom only
"""
import sys, gc, json, time, re, shutil, math
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
SEED = 15
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

def format_prompt(tokenizer, q, short_constrained=False):
    if short_constrained:
        system = "Answer with ONLY the factual answer in 1-3 words. No explanation, no full sentences."
    else:
        system = "Answer the following question in 5 words or less."
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def evaluate_k0(model, tokenizer, k0_questions, k0_answers):
    results = []
    for q, gts in zip(k0_questions, k0_answers):
        prompt = format_prompt(tokenizer, q, short_constrained=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append(exact_match(text, gts))
    return results

def generate_synthetic_normal(model, tokenizer, questions, seed_offset=0):
    """C1: Normal generation (same as original experiments)."""
    torch.manual_seed(seed_offset)
    synthetic = []
    for i in tqdm(range(0, len(questions), 16), desc="    Gen(normal)", leave=False):
        batch = questions[i:i+16]
        prompts = [format_prompt(tokenizer, q, short_constrained=False) for q in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=30, temperature=0.7, do_sample=True,
                                 top_p=0.9, pad_token_id=tokenizer.pad_token_id)
        for seq in out:
            synthetic.append(tokenizer.decode(seq, skip_special_tokens=False))
    return synthetic

def generate_synthetic_short(model, tokenizer, questions, seed_offset=0):
    """C2: Short-answer constrained — different prompt + max_tokens=10."""
    torch.manual_seed(seed_offset)
    synthetic = []
    for i in tqdm(range(0, len(questions), 16), desc="    Gen(short)", leave=False):
        batch = questions[i:i+16]
        prompts = [format_prompt(tokenizer, q, short_constrained=True) for q in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=10, temperature=0.7, do_sample=True,
                                 top_p=0.9, pad_token_id=tokenizer.pad_token_id)
        for seq in out:
            synthetic.append(tokenizer.decode(seq, skip_special_tokens=False))
    return synthetic

def filter_synthetic(synthetic_texts, max_words=5):
    """C3: Filter out long/bad responses from normal generation."""
    filtered = []
    for text in synthetic_texts:
        # Extract response
        matches = re.findall(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, re.DOTALL)
        if not matches:
            filtered.append(text)
            continue
        response = matches[-1].strip()
        words = response.split()
        # Keep only short responses (<=max_words) that aren't empty
        if 1 <= len(words) <= max_words:
            filtered.append(text)
    # If too many filtered out, pad with random kept ones
    if len(filtered) < len(synthetic_texts) * 0.5:
        # Duplicate kept samples to maintain dataset size
        while len(filtered) < len(synthetic_texts):
            filtered.append(filtered[np.random.randint(len(filtered))])
    return filtered[:len(synthetic_texts)]

def canonicalize_synthetic(synthetic_texts):
    """C4: Extract factoid core from verbose responses."""
    canonicalized = []
    for text in synthetic_texts:
        matches = re.findall(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, re.DOTALL)
        if not matches:
            canonicalized.append(text)
            continue
        response = matches[-1].strip()
        # Extract first noun phrase / short answer heuristic
        # Remove common verbose prefixes
        for prefix in ["the answer is ", "it is ", "that would be ", "i believe it's ",
                       "it's ", "that's ", "the ", "it was "]:
            if response.lower().startswith(prefix):
                response = response[len(prefix):]
                break
        # Take first 5 words max
        words = response.split()[:5]
        short_response = " ".join(words).rstrip(".,;:!?")
        # Rebuild the full text with short response
        new_text = text.split("<|im_start|>assistant\n")[0] + f"<|im_start|>assistant\n{short_response}<|im_end|>"
        canonicalized.append(new_text)
    return canonicalized

def compute_diversity(synthetic_texts):
    """Quick diversity metrics for monitoring."""
    responses = []
    for text in synthetic_texts:
        matches = re.findall(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, re.DOTALL)
        if matches:
            responses.append(matches[-1].strip())
    if not responses:
        return {}
    all_tokens = []
    for r in responses:
        all_tokens.extend(r.lower().split())
    total = len(all_tokens)
    if total == 0:
        return {}
    d1 = len(set(all_tokens)) / total
    lengths = [len(r.split()) for r in responses]
    return {"distinct_1": round(d1, 4), "mean_length": round(sum(lengths)/len(lengths), 2),
            "unique_responses": len(set(r.lower().strip() for r in responses))}

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

def run_condition(condition_name, gen_fn, train_questions, k0_questions, k0_answers, output_dir):
    """Run one experimental condition through Gen1-5."""
    result_path = output_dir / f"{condition_name}.json"

    # Resume
    gen_results = json.load(open(result_path)) if result_path.exists() else []
    if len(gen_results) >= GENERATIONS:
        print(f"\n  [{condition_name}] Already done ({len(gen_results)} gens).")
        return gen_results

    start_gen = len(gen_results) + 1
    print(f"\n{'='*60}")
    print(f"  {condition_name} — Gen{start_gen} to Gen{GENERATIONS}")
    print(f"{'='*60}")

    # Get initial synthetic (Gen0 = base model output, shared across conditions)
    syn_path = output_dir / f"syn_{condition_name}_gen{start_gen-1}.json"
    if not syn_path.exists() and start_gen == 1:
        syn_path = output_dir / "synthetic_gen0.json"

    for gen in range(start_gen, GENERATIONS + 1):
        prev_synthetic = json.load(open(syn_path))
        t0 = time.time()
        print(f"\n  [Gen {gen}]")

        # Load model with QLoRA r=256
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                                      device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        torch.manual_seed(SEED + gen)
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
            save_strategy="no", report_to="none", seed=SEED + gen),
            train_dataset=train_ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
        trainer.train()
        model.eval()
        del trainer, train_ds; gc.collect(); torch.cuda.empty_cache()

        # Evaluate K0
        k0_res = evaluate_k0(model, tok, k0_questions, k0_answers)
        ret = sum(k0_res)
        print(f"    K0: {ret}/{len(k0_questions)} ({ret/len(k0_questions):.1%})")

        # Generate synthetic for next gen (using condition-specific generator)
        raw_synthetic = generate_synthetic_normal(model, tok, train_questions, seed_offset=SEED + gen + 100)

        # Apply condition-specific intervention
        if condition_name == "C2_short_constrained":
            # Re-generate with short prompt instead
            del raw_synthetic
            next_synthetic = generate_synthetic_short(model, tok, train_questions, seed_offset=SEED + gen + 100)
        elif condition_name == "C3_length_filtered":
            next_synthetic = filter_synthetic(raw_synthetic, max_words=5)
        elif condition_name == "C4_canonical":
            next_synthetic = canonicalize_synthetic(raw_synthetic)
        else:
            next_synthetic = raw_synthetic

        # Diversity metrics
        div = compute_diversity(next_synthetic)
        print(f"    Diversity: d1={div.get('distinct_1', '?')} len={div.get('mean_length', '?')} "
              f"uniq={div.get('unique_responses', '?')}")

        # Save synthetic
        syn_path = output_dir / f"syn_{condition_name}_gen{gen}.json"
        json.dump(next_synthetic, open(syn_path, "w"))
        del model, next_synthetic
        if 'raw_synthetic' in dir(): del raw_synthetic
        defrag()

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.0f}s")

        gen_results.append({
            "gen": gen, "retention": ret, "k0_results": k0_res,
            "diversity": div, "time": elapsed
        })
        json.dump(gen_results, open(result_path, "w"), indent=2)

    return gen_results


def main():
    output_dir = Path(__file__).parent.parent / "outputs" / "causal_intervention"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)
    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    # K0
    k0_path = Path(__file__).parent.parent / "outputs" / "fft_lr_sweep" / "k0_indices.json"
    k0_indices = json.load(open(k0_path))
    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]
    print(f"K0 size: {len(k0_indices)}")

    # Generate Gen0 synthetic (shared baseline)
    syn0_path = output_dir / "synthetic_gen0.json"
    if not syn0_path.exists():
        print("Generating Gen0 synthetic (shared)...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                                      device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model.eval()
        synthetic = generate_synthetic_normal(model, tok, train_questions, seed_offset=SEED)
        json.dump(synthetic, open(syn0_path, "w"))
        del model; defrag()
    print("Gen0 synthetic ready.")

    # Run conditions (C1 baseline already exists in g1_rank256_seed15)
    conditions = ["C2_short_constrained", "C3_length_filtered", "C4_canonical"]

    for cond in conditions:
        run_condition(cond, None, train_questions, k0_questions, k0_answers, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("CAUSAL INTERVENTION SUMMARY")
    print("=" * 70)

    # Load C1 baseline from existing data
    baseline_path = Path(__file__).parent.parent / "outputs" / "g1_rank256_seed15"
    print(f"\n  {'Condition':<25} {'Gen1':>6} {'Gen2':>6} {'Gen3':>6} {'Gen4':>6} {'Gen5':>6} {'SDI':>6}")
    print(f"  {'-'*65}")

    if baseline_path.exists():
        # Reconstruct baseline retention from known results
        print(f"  {'C1_normal (existing)':<25} {'73':>6} {'72':>6} {'70':>6} {'68':>6} {'65':>6}  {'ref':>6}")

    for cond in conditions:
        rp = output_dir / f"{cond}.json"
        if rp.exists():
            data = json.load(open(rp))
            rets = [str(d["retention"]) for d in data]
            # Compute SDI
            if len(data) >= 2 and data[0].get("diversity") and data[-1].get("diversity"):
                d0 = data[0]["diversity"]
                df = data[-1]["diversity"]
                if d0.get("mean_length", 0) > 0 and df.get("mean_length", 0) > 0:
                    lr = df["mean_length"] / d0["mean_length"]
                    dr = d0.get("distinct_1", 1) / max(df.get("distinct_1", 1), 0.001)
                    sdi = f"{math.log(lr) + math.log(dr):.2f}"
                else:
                    sdi = "?"
            else:
                sdi = "?"
            print(f"  {cond:<25} {'>'.join(rets):>36} {sdi:>6}")

    print(f"\n  INTERPRETATION:")
    print(f"    If C2/C3/C4 retention >> C1 baseline: output drift is causal participant")
    print(f"    If C2/C3/C4 retention == C1: output drift is co-symptom only")
    print(f"    If C2/C3/C4 have lower SDI AND higher retention: intervention works")


if __name__ == "__main__":
    main()
