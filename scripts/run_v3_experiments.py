"""
Run all new experiments required for v3 (Carlos Review 2).

Experiments needed:
1. Gemma 3: extend existing runs to Gen10 (currently only Gen5)
   - r=4 seeds 15, 137, 256, 42, 77
   - r=16 seeds 15, 137, 256, 42, 77
2. Gemma 4 E2B: extend to Gen10 with additional rank
   - r=4 seed 15 (extend from Gen5 to Gen10)
   - r=16 seed 15 (extend from Gen5 to Gen10)
   - r=64 seed 15 (new, to have dose-response curve)

All save per-generation results to JSON, skip already-completed generations.
Output format matches existing experiments for graph generation.

Run on Athena:
  tmux new -s v3exp
  uv run python scripts/run_v3_experiments.py
"""
import sys, gc, json, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset

# ============================================================
# CONFIG
# ============================================================
TRAIN_SIZE = 2000
EVAL_SIZE = 200
TARGET_GENERATIONS = 10
LR = 1e-5
OUTPUT_BASE = Path("outputs")

# All experiments to run
EXPERIMENTS = [
    # Gemma 3: extend to Gen10
    {"model": "google/gemma-3-1b-it", "rank": 4, "seed": 15, "label": "gemma3"},
    {"model": "google/gemma-3-1b-it", "rank": 4, "seed": 137, "label": "gemma3"},
    {"model": "google/gemma-3-1b-it", "rank": 4, "seed": 256, "label": "gemma3"},
    {"model": "google/gemma-3-1b-it", "rank": 4, "seed": 42, "label": "gemma3"},
    {"model": "google/gemma-3-1b-it", "rank": 4, "seed": 77, "label": "gemma3"},
    {"model": "google/gemma-3-1b-it", "rank": 16, "seed": 15, "label": "gemma3"},
    {"model": "google/gemma-3-1b-it", "rank": 16, "seed": 137, "label": "gemma3"},
    {"model": "google/gemma-3-1b-it", "rank": 16, "seed": 256, "label": "gemma3"},
    {"model": "google/gemma-3-1b-it", "rank": 16, "seed": 42, "label": "gemma3"},
    {"model": "google/gemma-3-1b-it", "rank": 16, "seed": 77, "label": "gemma3"},
    # Gemma 4 E2B: extend to Gen10 + new rank
    {"model": "google/gemma-4-E2B-it", "rank": 4, "seed": 15, "label": "gemma4"},
    {"model": "google/gemma-4-E2B-it", "rank": 16, "seed": 15, "label": "gemma4"},
    {"model": "google/gemma-4-E2B-it", "rank": 64, "seed": 15, "label": "gemma4"},
]

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


# ============================================================
# UTILITIES
# ============================================================
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


def load_model(model_name):
    """Load model with 4-bit quantization."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    # Gemma 4 needs patching
    if "gemma-4" in model_name.lower():
        from gemma4_peft_patch import patch_gemma4_for_peft
        model = patch_gemma4_for_peft(model)

    return model, tokenizer


def format_prompt(tokenizer, q):
    msgs = [{"role": "user", "content": f"Answer the following question in 5 words or less.\n{q}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def evaluate_k0(model, tokenizer, k0_questions, k0_answers):
    """Evaluate K0 retention. Returns list of booleans + raw responses."""
    results = []
    responses = []
    for q, gts in zip(k0_questions, k0_answers):
        prompt = format_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append(exact_match(text, gts))
        responses.append(text)
    return results, responses


def generate_synthetic(model, tokenizer, questions, seed_offset=0):
    """Generate synthetic responses in batches of 8."""
    torch.manual_seed(seed_offset)
    synthetic = []
    raw_responses = []
    for i in range(0, len(questions), 8):
        batch = questions[i:i+8]
        prompts = [format_prompt(tokenizer, q) for q in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, temperature=0.7, do_sample=True, top_p=0.9)
        for j, seq in enumerate(out):
            full_text = tokenizer.decode(seq, skip_special_tokens=False)
            synthetic.append(full_text)
            # Also store just the response for distributional analysis
            resp_text = tokenizer.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            raw_responses.append(resp_text)
    return synthetic, raw_responses


def compute_lora_spectrum(model):
    """Compute effective rank and Frobenius norm of LoRA updates."""
    ranks, f_norms = [], []
    params_dict = dict(model.named_parameters())
    for name, param in model.named_parameters():
        if "lora_A" in name and param.requires_grad:
            b_name = name.replace("lora_A", "lora_B")
            b_param = params_dict.get(b_name)
            if b_param is not None:
                AB = b_param.data.float().cpu() @ param.data.float().cpu()
                svs = torch.linalg.svdvals(AB)
                svs_n = svs / (svs.sum() + 1e-10)
                ranks.append(torch.exp(-(svs_n * torch.log(svs_n + 1e-10)).sum()).item())
                f_norms.append(torch.norm(AB, 'fro').item())
    return {"eff_rank": float(np.mean(ranks)), "frobenius": float(np.mean(f_norms))}


def compute_distributional_metrics(responses):
    """Compute output-distribution diagnostics for a set of responses."""
    if not responses:
        return {}

    # Mean response length (words)
    lengths = [len(r.split()) for r in responses]
    mean_length = float(np.mean(lengths))

    # Distinct-1 and Distinct-2
    all_unigrams = []
    all_bigrams = []
    for r in responses:
        words = r.lower().split()
        all_unigrams.extend(words)
        all_bigrams.extend(zip(words[:-1], words[1:]))
    distinct_1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)

    # Stopword ratio
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "could",
                 "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                 "on", "with", "at", "by", "from", "it", "this", "that", "and", "or",
                 "but", "if", "not", "no", "so", "as"}
    total_words = len(all_unigrams)
    stop_count = sum(1 for w in all_unigrams if w in stopwords)
    stopword_ratio = stop_count / max(total_words, 1)

    return {
        "mean_length_words": round(mean_length, 2),
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "stopword_ratio": round(stopword_ratio, 4),
        "total_responses": len(responses),
    }


# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================
def run_experiment(exp_config, train_questions, k0_indices, k0_questions, k0_answers):
    """Run a single experiment configuration through TARGET_GENERATIONS generations."""
    model_name = exp_config["model"]
    rank = exp_config["rank"]
    seed = exp_config["seed"]
    label = exp_config["label"]

    output_dir = OUTPUT_BASE / f"v3_{label}_rank{rank}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"

    print(f"\n{'='*70}")
    print(f"  {label.upper()} — r={rank}, seed={seed} → {output_dir.name}")
    print(f"{'='*70}")

    # Load existing results if any
    existing_gens = []
    last_synthetic = None
    if result_path.exists():
        existing_data = json.loads(result_path.read_text())
        existing_gens = existing_data.get("generations", [])
        max_gen_done = max((g["gen"] for g in existing_gens), default=0)

        if max_gen_done >= TARGET_GENERATIONS:
            print(f"  Already complete ({max_gen_done} gens). Skipping.")
            return

        # Need to regenerate synthetic from the last completed generation
        # Load the last synthetic file if saved
        synth_path = output_dir / f"synthetic_gen{max_gen_done}.json"
        if synth_path.exists():
            last_synthetic = json.loads(synth_path.read_text())
            print(f"  Resuming from Gen{max_gen_done+1} (loaded synthetic_gen{max_gen_done}.json)")
        else:
            # Must regenerate from scratch
            print(f"  Found {max_gen_done} gens but no synthetic file. Restarting from Gen0.")
            existing_gens = []
            max_gen_done = 0
    else:
        max_gen_done = 0

    # Gen 0: generate initial synthetic data
    if last_synthetic is None:
        print(f"  Generating initial synthetic data (Gen0)...")
        model, tokenizer = load_model(model_name)
        synthetic_texts, raw_responses = generate_synthetic(model, tokenizer, train_questions, seed_offset=seed)

        # Save Gen0 distributional metrics
        gen0_metrics = compute_distributional_metrics(raw_responses)
        gen0_path = output_dir / "gen0_metrics.json"
        gen0_path.write_text(json.dumps(gen0_metrics, indent=2))

        # Save synthetic for resumability
        synth_path = output_dir / "synthetic_gen0.json"
        synth_path.write_text(json.dumps(synthetic_texts))
        last_synthetic = synthetic_texts

        del model; defrag()
        max_gen_done = 0

    # Recursive training loop
    gen_results = list(existing_gens)
    for gen in range(max_gen_done + 1, TARGET_GENERATIONS + 1):
        t0 = time.time()
        print(f"  [Gen {gen}/{TARGET_GENERATIONS}] r={rank}, seed={seed}")

        try:
            model, tokenizer = load_model(model_name)

            # Create LoRA adapter
            torch.manual_seed(seed + gen)
            lora_config = LoraConfig(
                r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
            )
            model.enable_input_require_grads()
            peft_model = get_peft_model(model, lora_config)

            # Prepare training data
            def tok_fn(ex):
                return tokenizer(ex["text"], truncation=True, max_length=256, padding="max_length")
            train_ds = Dataset.from_dict({"text": last_synthetic})
            train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
            train_ds.set_format("torch")

            # Train
            trainer = Trainer(
                model=peft_model,
                args=TrainingArguments(
                    output_dir=str(output_dir / "tmp"),
                    num_train_epochs=2,
                    per_device_train_batch_size=8,
                    gradient_accumulation_steps=2,
                    learning_rate=LR,
                    bf16=True,
                    logging_steps=9999,
                    save_strategy="no",
                    report_to="none",
                    seed=seed + gen,
                ),
                train_dataset=train_ds,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            trainer.train()
            peft_model.eval()

            # Evaluate
            spectrum = compute_lora_spectrum(peft_model)
            k0_res, k0_responses = evaluate_k0(peft_model, tokenizer, k0_questions, k0_answers)
            retention = sum(k0_res)

            # Generate next synthetic + distributional metrics
            synthetic_texts, raw_responses = generate_synthetic(
                peft_model, tokenizer, train_questions, seed_offset=seed + gen + 100
            )
            dist_metrics = compute_distributional_metrics(raw_responses)

            elapsed = time.time() - t0
            print(f"    K0: {retention}/{len(k0_indices)} ({retention/len(k0_indices):.1%}) | "
                  f"erank={spectrum['eff_rank']:.2f} | len={dist_metrics['mean_length_words']:.1f}w [{elapsed:.0f}s]")

            # Save generation result
            gen_data = {
                "gen": gen,
                "retention": retention,
                "k0_size": len(k0_indices),
                "eff_rank": spectrum["eff_rank"],
                "frobenius": spectrum["frobenius"],
                "distributional": dist_metrics,
                "k0_per_item": k0_res,
                "time_sec": round(elapsed, 1),
            }
            gen_results.append(gen_data)

            # Save synthetic for next gen and resumability
            synth_path = output_dir / f"synthetic_gen{gen}.json"
            synth_path.write_text(json.dumps(synthetic_texts))
            last_synthetic = synthetic_texts

            # Save cumulative results after each generation
            full_result = {
                "model": model_name,
                "rank": rank,
                "seed": seed,
                "k0_size": len(k0_indices),
                "target_generations": TARGET_GENERATIONS,
                "generations": gen_results,
            }
            result_path.write_text(json.dumps(full_result, indent=2))

            del peft_model, trainer, model; defrag()

        except Exception as e:
            print(f"    ❌ ERROR at Gen {gen}: {e}")
            print(f"    Saving progress and moving to next experiment.")
            # Save what we have
            if gen_results:
                full_result = {
                    "model": model_name,
                    "rank": rank,
                    "seed": seed,
                    "k0_size": len(k0_indices),
                    "target_generations": TARGET_GENERATIONS,
                    "generations": gen_results,
                    "error": f"Failed at gen {gen}: {str(e)}",
                }
                result_path.write_text(json.dumps(full_result, indent=2))
            defrag()
            break

    print(f"  ✓ Completed: {len(gen_results)} generations saved to {result_path.name}")


def main():
    print("=" * 70)
    print("  V3 EXPERIMENTS — Extending Gemma 3 to Gen10 + E2B expansion")
    print("=" * 70)
    print(f"  Total experiments: {len(EXPERIMENTS)}")
    print(f"  Target generations: {TARGET_GENERATIONS}")
    print()

    # Load dataset (same as all other experiments)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)
    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]]
                    for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    # We need K0 per model family (different base models have different K0)
    k0_cache = {}

    for exp in EXPERIMENTS:
        model_name = exp["model"]

        # Establish K0 for this model if not cached
        if model_name not in k0_cache:
            print(f"\n  Establishing K0 for {model_name}...")
            model, tokenizer = load_model(model_name)
            gen0_results, _ = evaluate_k0(model, tokenizer, eval_questions, eval_answers)
            k0_indices = [i for i, r in enumerate(gen0_results) if r]
            k0_questions = [eval_questions[i] for i in k0_indices]
            k0_answers = [eval_answers[i] for i in k0_indices]
            k0_cache[model_name] = (k0_indices, k0_questions, k0_answers)
            print(f"  K0 for {model_name}: {len(k0_indices)}/{EVAL_SIZE}")
            del model; defrag()

        k0_indices, k0_questions, k0_answers = k0_cache[model_name]

        try:
            run_experiment(exp, train_questions, k0_indices, k0_questions, k0_answers)
        except Exception as e:
            print(f"\n  ⚠️ FATAL ERROR for {exp['label']} r={exp['rank']} seed={exp['seed']}: {e}")
            print(f"  Continuing with next experiment...\n")
            defrag()
            continue

    # Final summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for exp in EXPERIMENTS:
        label = exp["label"]
        rank = exp["rank"]
        seed = exp["seed"]
        result_path = OUTPUT_BASE / f"v3_{label}_rank{rank}_seed{seed}" / "results.json"
        if result_path.exists():
            data = json.loads(result_path.read_text())
            gens = data.get("generations", [])
            last_gen = gens[-1] if gens else None
            if last_gen:
                k0 = data["k0_size"]
                ret = last_gen["retention"]
                print(f"  {label} r={rank} seed={seed}: Gen{last_gen['gen']} → "
                      f"{ret}/{k0} ({ret/k0*100:.1f}%) erank={last_gen['eff_rank']:.2f}")
            else:
                print(f"  {label} r={rank} seed={seed}: NO DATA")
        else:
            print(f"  {label} r={rank} seed={seed}: NOT STARTED")


if __name__ == "__main__":
    main()
