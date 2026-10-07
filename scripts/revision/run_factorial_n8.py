#!/usr/bin/env python3
"""
run_factorial_n8.py — Branched 2x2 factorial, N=8 seeds, corpus-per-seed

Four changes vs run_factorial_p4a.py:
  1. 8 seeds (df=7 instead of df=2)
  2. Real corpus subsampled per seed (IC bounds corpus effect, not just RNG)
  3. Qwen uses shuffled split (dataset_seed=15, same as Gemma)
  4. evaluate_with_raw_predictions in ALL generations (scorer auditable)

Usage:
  python run_factorial_n8.py --backbone gemma --seed 42 --gpu 1
  python run_factorial_n8.py --backbone qwen  --seed 42 --gpu 1
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

# GPU pinning before torch import
def _pin_gpu(argv):
    # CUDA_DEVICE_ORDER ensures nvidia-smi index == CUDA device index.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    for i, a in enumerate(argv):
        if a == "--gpu" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]
            return int(argv[i + 1])
        elif a.startswith("--gpu="):
            val = a.split("=", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = val
            return int(val)
    return None

PHYSICAL_GPU = _pin_gpu(sys.argv)

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset, load_dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from common import (
    BACKBONES, EPOCHS, LR, MAX_LENGTH, TRAIN_SIZE, DATASET, DATASET_SEED,
    evaluate_k0, generate_synthetic, load_k0, load_trivia,
    get_batch_size, defrag, format_prompt, load_tokenizer
)
from io_utils import load_json, save_json, clean_stale_tmp

# ── constants ────────────────────────────────────────────────────────────────
SEEDS = [42, 137, 256, 15, 77, 99, 101, 202]
REAL_CORPUS_SIZE = 2000   # subsample size — same as existing runs


# ── corpus per seed ───────────────────────────────────────────────────────────

def build_real_corpus_for_seed(seed, train_q, train_a, tokenizer, cfg):
    """
    Subsample REAL corpus from train pool using seed-specific RNG.
    CRITICAL: each seed gets a distinct corpus so seeds are genuine replications,
    not just training-RNG replicates under a fixed corpus.
    """
    pool_size = min(len(train_q), len(train_a))
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(pool_size, size=REAL_CORPUS_SIZE, replace=False))

    corpus = []
    for i in idx:
        q = train_q[i]
        a = train_a[i]
        answer = a[0] if isinstance(a, list) and a else str(a)
        prompt = format_prompt(tokenizer, q, cfg)
        corpus.append(prompt + " " + answer)

    h = hashlib.md5(json.dumps(corpus).encode()).hexdigest()[:8]
    print(f"    Real corpus seed={seed}: {len(corpus)} ex, hash={h}")
    return corpus, h


# ── scorer helpers ────────────────────────────────────────────────────────────

def normalize(s):
    import re
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()

def substring_score(pred, aliases):
    p = normalize(pred)
    for a in aliases:
        a_n = normalize(a)
        if not a_n: continue
        if a_n in p or p in a_n:
            return True
    return False

def token_f1(pred, aliases):
    p_toks = set(normalize(pred).split())
    best = 0.0
    for a in aliases:
        a_toks = set(normalize(a).split())
        if not a_toks: continue
        inter = len(p_toks & a_toks)
        if inter == 0: continue
        prec = inter / len(p_toks) if p_toks else 0
        rec  = inter / len(a_toks)
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        best = max(best, f1)
    return best


# ── evaluation with raw predictions ──────────────────────────────────────────

def evaluate_with_raw(model, tokenizer, k0_q, k0_a, cfg):
    """
    Full evaluation returning per-item raw predictions.
    Mudança 4: usado em TODAS as gerações.
    """
    model.eval()
    device = next(model.parameters()).device
    preds = []

    for i, (q, aliases) in enumerate(zip(k0_q, k0_a)):
        prompt = format_prompt(tokenizer, q, cfg)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
        ilen = inputs["input_ids"].shape[1]
        olen = out.shape[1] - ilen
        response = tokenizer.decode(out[0][ilen:], skip_special_tokens=True).strip()

        # Classify disjuncts for second-disjunct audit
        p_n = normalize(response)
        d1 = d2 = False
        for a in aliases:
            a_n = normalize(a)
            if not a_n: continue
            if a_n in p_n: d1 = True
            if p_n in a_n: d2 = True

        preds.append({
            "index": i, "question": q,
            "gold_aliases": aliases[:5],  # cap to save space
            "prediction": response,
            "output_tokens": olen,
            "truncated": olen >= 20,
            "substring_correct": d1 or d2,
            "disjunct1_only": d1 and not d2,  # alias-in-pred
            "disjunct2_only": d2 and not d1,  # pred-in-alias (short pred rewarded)
            "token_f1": round(token_f1(response, aliases), 4),
        })

    retention = sum(1 for p in preds if p["substring_correct"])
    model.train()
    return {"retention": retention, "raw_predictions": preds}


# ── checkpoint helpers ────────────────────────────────────────────────────────

def capture_rng():
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s

def restore_rng(s):
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    if torch.cuda.is_available() and "cuda" in s:
        torch.cuda.set_rng_state_all(s["cuda"])

def save_ckpt(out_dir, gen, model, corpus, rng, results):
    d = out_dir / f"checkpoint_gen{gen}"
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d / "adapter")
    save_json(corpus, d / "corpus.json")
    torch.save(rng, d / "rng_states.pt")
    save_json(results, d / "results.json")
    print(f"    Checkpoint saved: {d}")
    return d

def load_ckpt(ckpt_dir, base_model):
    model = PeftModel.from_pretrained(base_model, ckpt_dir / "adapter", is_trainable=True)
    model.train()
    corpus = load_json(ckpt_dir / "corpus.json", expect=list)
    rng = torch.load(ckpt_dir / "rng_states.pt", weights_only=False)
    results = load_json(ckpt_dir / "results.json", expect=list)
    return model, corpus, rng, results


# ── training ──────────────────────────────────────────────────────────────────

def train_gen(model, tokenizer, corpus, cfg, gen, work_dir):
    clean_stale_tmp(work_dir)
    tmp = work_dir / f"tmp_gen{gen}"
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    def tok(ex):
        return tokenizer(ex["text"], truncation=True,
                         max_length=MAX_LENGTH, padding=False)

    ds = Dataset.from_dict({"text": corpus}).map(tok, remove_columns=["text"])
    bs = cfg.get("train_batch", 4)
    ga = cfg.get("train_accum", 4)

    args = TrainingArguments(
        output_dir=str(tmp), num_train_epochs=EPOCHS,
        per_device_train_batch_size=bs, gradient_accumulation_steps=ga,
        learning_rate=LR, warmup_steps=0, logging_steps=50,
        save_strategy="no", report_to="none", fp16=True,
        dataloader_num_workers=0, remove_unused_columns=False,
    )
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    ).train()

    if tmp.exists(): shutil.rmtree(tmp)
    gc.collect(); torch.cuda.empty_cache()
    return model


# ── load data ─────────────────────────────────────────────────────────────────

def load_data(backbone):
    """
    Mudança 3: Qwen usa split shuffled (dataset_seed=15) igual ao Gemma.
    FIX: train_q usa dataset completo (80000) para corpus real distinto por seed.
    """
    print(f"  Loading TriviaQA (shuffled split, dataset_seed={DATASET_SEED})...")
    # Load full dataset for train pool (80000 items)
    ds = load_dataset(*DATASET, split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=DATASET_SEED)
    n_train = min(len(ds), 80000)
    train_q = [ds[i]["question"] for i in range(n_train)]
    train_a = [
        ds[i]["answer"]["aliases"] + [ds[i]["answer"]["value"]]
        for i in range(n_train)
    ]
    # eval from load_trivia (uses TRAIN_SIZE offset for eval partition)
    _, eval_q, eval_a = load_trivia()
    print(f"  train_q={len(train_q)}, train_a={len(train_a)}, eval_q={len(eval_q)}")
    return train_q, train_a, eval_q, eval_a


# ── shared prefix (Phase A1) ──────────────────────────────────────────────────

def run_a1(args, cfg, out_dir, train_q, k0_q, k0_a):
    print("\n" + "="*70)
    print("  PHASE A1: Shared Prefix (Gen 0-2)")
    print("="*70)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"], quantization_config=bnb, device_map="auto",
        trust_remote_code=True)
    tok = load_tokenizer(cfg)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    lora_cfg = LoraConfig(r=cfg["rank"], lora_alpha=cfg["rank"]*2,
                           target_modules=["q_proj","v_proj"], lora_dropout=0.05,
                           bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora_cfg)

    corpus = generate_synthetic(model, tok, train_q[:TRAIN_SIZE], cfg)
    results = []

    for gen in range(3):
        print(f"\n  === Gen {gen} ===")
        # Mudança 4: raw predictions em TODAS as gerações
        raw = evaluate_with_raw(model, tok, k0_q, k0_a, cfg)
        ret = raw["retention"]
        results.append({
            "gen": gen, "retention": raw["retention"],
            "retention_pct": raw["retention"] / len(k0_q),
            "k0_size": len(k0_q),
            "raw_predictions": raw["raw_predictions"],
            "corpus_hash": hashlib.md5(
                json.dumps(corpus).encode()).hexdigest()[:8],
        })
        print(f"    K0: {ret}/{len(k0_q)} = {100*ret/len(k0_q):.2f}%")
        model = train_gen(model, tok, corpus, cfg, gen, out_dir)
        corpus = generate_synthetic(model, tok, train_q[:TRAIN_SIZE], cfg)

    rng = capture_rng()
    ckpt = save_ckpt(out_dir, 2, model, corpus, rng, results)
    save_json(results, out_dir / "results_a1.json")

    del model, base
    gc.collect(); torch.cuda.empty_cache()
    return ckpt, results


# ── branch (Phase B + A2) ─────────────────────────────────────────────────────

def run_branch(args, cfg, ckpt_dir, branch, adapter_mode, corpus_mode,
               out_dir, train_q, real_corpus, k0_q, k0_a):
    print(f"\n{'='*70}\n  BRANCH: {branch} ({adapter_mode} + {corpus_mode})\n{'='*70}")

    bdir = out_dir / f"branch_{branch}"
    bdir.mkdir(parents=True, exist_ok=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"], quantization_config=bnb, device_map="auto",
        trust_remote_code=True)
    tok = load_tokenizer(cfg)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model, corpus, rng, a1_results = load_ckpt(ckpt_dir, base)
    restore_rng(rng)

    if adapter_mode == "reset":
        print("  Resetting adapter...")
        b = model.get_base_model()
        del model; gc.collect(); torch.cuda.empty_cache()
        lora_cfg = LoraConfig(r=cfg["rank"], lora_alpha=cfg["rank"]*2,
                               target_modules=["q_proj","v_proj"], lora_dropout=0.05,
                               bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(b, lora_cfg)

    results = []

    # Phase B: Gen 3-6
    print("\n  --- Phase B ---")
    for gen in range(3, 7):
        print(f"\n  === Gen {gen} ===")
        raw = evaluate_with_raw(model, tok, k0_q, k0_a, cfg)
        ret = raw["retention"]
        train_corpus = real_corpus if corpus_mode == "real" else corpus
        results.append({
            "gen": gen, "phase": "B", "branch": branch,
            "adapter_mode": adapter_mode, "corpus_mode": corpus_mode,
            "retention": ret,
            "retention_pct": ret / len(k0_q),
            "k0_size": len(k0_q),
            "raw_predictions": raw["raw_predictions"],
            "corpus_hash": hashlib.md5(
                json.dumps(train_corpus).encode()).hexdigest()[:8],
        })
        print(f"    K0: {ret}/{len(k0_q)} = {100*ret/len(k0_q):.2f}%"
              f"  corpus={'real' if corpus_mode=='real' else 'synth'}")
        model = train_gen(model, tok, train_corpus, cfg, gen, bdir)
        corpus = generate_synthetic(model, tok, train_q[:TRAIN_SIZE], cfg)

    # Phase A2: Gen 7-12
    print("\n  --- Phase A2 ---")
    for gen in range(7, 13):
        print(f"\n  === Gen {gen} ===")
        raw = evaluate_with_raw(model, tok, k0_q, k0_a, cfg)
        ret = raw["retention"]
        results.append({
            "gen": gen, "phase": "A2", "branch": branch,
            "adapter_mode": adapter_mode, "corpus_mode": corpus_mode,
            "retention": ret,
            "retention_pct": ret / len(k0_q),
            "k0_size": len(k0_q),
            "raw_predictions": raw["raw_predictions"],
            "corpus_hash": hashlib.md5(
                json.dumps(corpus).encode()).hexdigest()[:8],
        })
        print(f"    K0: {ret}/{len(k0_q)} = {100*ret/len(k0_q):.2f}%  corpus=synth")
        model = train_gen(model, tok, corpus, cfg, gen, bdir)
        corpus = generate_synthetic(model, tok, train_q[:TRAIN_SIZE], cfg)

    save_json(results, bdir / "results.json")
    del model, base
    gc.collect(); torch.cuda.empty_cache()
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--branches", type=str, default="all")
    args = parser.parse_args()

    assert args.seed in SEEDS, f"seed {args.seed} not in {SEEDS}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = BACKBONES[args.backbone].copy()
    cfg["rank"] = 16  # matched rank for both backbones

    if args.output is None:
        args.output = f"outputs/factorial_n8_{args.backbone}"
    out_dir = Path(args.output) / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(f"  Factorial N=8 — backbone={args.backbone}, seed={args.seed}")
    print(f"  GPU: {PHYSICAL_GPU}  |  rank: {cfg['rank']}")
    print(f"  Output: {out_dir}")
    print("="*70)

    # Load data — Mudança 3: ambos os backbones usam split shuffled
    train_q, train_a, eval_q, eval_a = load_data(args.backbone)

    # Load K0
    k0_q, k0_a, k0_meta = load_k0(args.backbone, eval_q, eval_a)
    print(f"  K0: {len(k0_q)} items")

    # Build tokenizer for corpus construction
    tok_tmp = load_tokenizer(cfg)
    if tok_tmp.pad_token is None: tok_tmp.pad_token = tok_tmp.eos_token

    # Mudança 2: corpus real subamostrado por seed
    real_corpus, corpus_hash = build_real_corpus_for_seed(
        args.seed, train_q, train_a, tok_tmp, cfg)
    del tok_tmp

    # Save corpus metadata
    save_json({
        "seed": args.seed, "backbone": args.backbone,
        "size": len(real_corpus), "hash": corpus_hash,
        "pool_size": len(train_q),
        "timestamp": datetime.now().isoformat(),
        "note": "Subsampled per-seed for genuine replication"
    }, out_dir / "real_corpus_meta.json")

    # Phase A1
    ckpt_dir = out_dir / "checkpoint_gen2"
    if ckpt_dir.exists():
        print(f"\n  Resuming from checkpoint: {ckpt_dir}")
        a1_results = load_json(out_dir / "results_a1.json", expect=list)
    else:
        ckpt_dir, a1_results = run_a1(args, cfg, out_dir, train_q, k0_q, k0_a)

    # Branches
    all_branches = [
        ("C-S", "continued", "synthetic"),
        ("C-R", "continued", "real"),
        ("R-S", "reset",     "synthetic"),
        ("R-R", "reset",     "real"),
    ]
    if args.branches != "all":
        req = set(args.branches.split(","))
        all_branches = [(n,a,c) for n,a,c in all_branches if n in req]

    all_results = {"a1": a1_results, "branches": {}, "k0_meta": k0_meta}

    for bname, amode, cmode in all_branches:
        bpath = out_dir / f"branch_{bname}" / "results.json"
        if bpath.exists():
            print(f"\n  Branch {bname}: loading existing results...")
            all_results["branches"][bname] = load_json(bpath, expect=list)
        else:
            all_results["branches"][bname] = run_branch(
                args, cfg, ckpt_dir, bname, amode, cmode,
                out_dir, train_q, real_corpus, k0_q, k0_a)

    save_json(all_results, out_dir / "results_all.json")

    # Summary
    print("\n" + "="*70 + "\n  SUMMARY Gen7\n" + "="*70)
    vals = {}
    for bn in ["C-S","C-R","R-S","R-R"]:
        if bn in all_results["branches"]:
            r7 = next((r for r in all_results["branches"][bn] if r["gen"]==7), None)
            if r7:
                pct = r7["retention"] / r7["k0_size"] * 100
                vals[bn] = pct
                print(f"  {bn}: {r7['retention']}/{r7['k0_size']} = {pct:.2f}%")

    if len(vals) == 4:
        inter = ((vals["R-R"] - vals["R-S"]) - (vals["C-R"] - vals["C-S"]))
        print(f"\n  Interaction (R-R−R-S)−(C-R−C-S) = {inter:+.2f}pp")
        print(f"  {'✓ NEGATIVE' if inter < 0 else '✗ POSITIVE'}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
