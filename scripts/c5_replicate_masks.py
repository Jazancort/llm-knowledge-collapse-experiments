"""C5 replication: 2 additional random masks to confirm boundary sensitivity.

Same as validate_intervention.py C5, but with different random seeds for downsampling.
If all 3 masks give ~71-73/78: "any ~5% reduction stabilizes" is confirmed.
If varies wildly: effect is mask-dependent.

Run on Athena:
  uv run python scripts/c5_replicate_masks.py
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
TARGET_TOKENS = 54500
MASKS = [42, 99]  # Two new random seeds for downsampling

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

def downsample(synthetic_texts, target_tokens, mask_seed):
    np.random.seed(mask_seed)
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
    gc.collect(); torch.cuda.empty_cache(); gc.collect()
    try:
        import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass
    time.sleep(2)

def get_batch_size():
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb >= 20: return 4, 4
    elif total_gb >= 12: return 2, 8
    else: return 1, 16

def main():
    output_dir = Path(__file__).parent.parent / "outputs" / "c5_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=15)
    train_questions = [ds[i]["question"] for i in range(TRAIN_SIZE)]
    eval_questions = [ds[i]["question"] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]
    eval_answers = [ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]] for i in range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE)]

    k0_indices = json.load(open(Path(__file__).parent.parent / "outputs" / "fft_lr_sweep" / "k0_indices.json"))
    k0_questions = [eval_questions[i] for i in k0_indices]
    k0_answers = [eval_answers[i] for i in k0_indices]
    print(f"K0 size: {len(k0_indices)}")

    # Gen0 synthetic (shared)
    syn0_path = output_dir / "synthetic_gen0.json"
    if not syn0_path.exists():
        print("Generating Gen0...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                                      device_map="auto", torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"; model.eval()
        syn = generate_synthetic(model, tok, train_questions, seed_offset=SEED)
        json.dump(syn, open(syn0_path, "w"))
        del model; defrag()

    for mask_seed in MASKS:
        key = f"C5_mask{mask_seed}"
        result_path = output_dir / f"{key}.json"
        gen_results = json.load(open(result_path)) if result_path.exists() else []
        if len(gen_results) >= GENERATIONS:
            print(f"\n  [{key}] Done.")
            continue

        start_gen = len(gen_results) + 1
        print(f"\n{'='*50}\n  {key} — Gen{start_gen} to Gen{GENERATIONS}\n{'='*50}")

        syn_path = output_dir / f"syn_{key}_gen{start_gen-1}.json"
        if not syn_path.exists():
            syn_path = syn0_path

        for gen in range(start_gen, GENERATIONS + 1):
            prev = json.load(open(syn_path))
            t0 = time.time()
            print(f"\n  [Gen {gen}] ({len(prev)} examples)")

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

            def tok_fn(ex):
                return tok(ex["text"], truncation=True, max_length=256, padding="max_length")
            train_ds = Dataset.from_dict({"text": prev})
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

            k0_res = evaluate_k0(model, tok, k0_questions, k0_answers)
            ret = sum(k0_res)
            print(f"    K0: {ret}/{len(k0_questions)} ({ret/len(k0_questions):.1%})")

            raw_syn = generate_synthetic(model, tok, train_questions, seed_offset=SEED + gen + 100)
            processed = downsample(raw_syn, TARGET_TOKENS, mask_seed + gen)
            syn_path = output_dir / f"syn_{key}_gen{gen}.json"
            json.dump(processed, open(syn_path, "w"))
            del model, raw_syn, processed; defrag()

            elapsed = time.time() - t0
            print(f"    Time: {elapsed:.0f}s")
            gen_results.append({"gen": gen, "retention": ret, "time": elapsed})
            json.dump(gen_results, open(result_path, "w"), indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("C5 MASK REPLICATION SUMMARY")
    print("=" * 60)
    print(f"  C5 mask=15 (original): 74 73 74 72 72")
    for mask_seed in MASKS:
        rp = output_dir / f"C5_mask{mask_seed}.json"
        if rp.exists():
            data = json.load(open(rp))
            rets = [str(d["retention"]) for d in data]
            print(f"  C5 mask={mask_seed}:          {' '.join(rets)}")
    print(f"\n  C1 normal:             73 72 70 68 65")
    print(f"  C3 filtered:           74 74 71 71 73")
    print(f"\n  If all masks ~71-73: boundary sensitivity confirmed.")
    print(f"  If varies >5pp: mask-dependent (some removals better than others).")

if __name__ == "__main__":
    main()
