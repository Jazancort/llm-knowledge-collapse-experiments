#!/usr/bin/env python3
"""g2_pilot_sheared.py — G2 gate for Sheared-LLaMA-1.3B.

Runs Gen0–3 on seed=15, C-S arm only (continued adapter + synthetic corpus).
Measures R3 = retention at Gen3 to decide onset rule (G2 gate).

Decision rule (pre-registered):
  R3 >= 0.40  -> PRIMARY protocol: onset Gen3 (same as Llama)
  0.20 <= R3 < 0.40 -> FALLBACK: onset Gen2
  R3 < 0.20  -> STOP

Outputs:
  outputs/sheared_g2/g2_result.json  (R3, Gen2 checkpoint hash, timing)
  outputs/sheared_g2/gen2_adapter/   (adapter weights for pre-registration)

Usage:
  export CUDA_VISIBLE_DEVICES=1
  uv run python scripts/revision/g2_pilot_sheared.py --gpu 0
"""
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path

def _pin_gpu(argv):
    for i, a in enumerate(argv):
        if a == "--gpu" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]; return int(argv[i + 1])
        elif a.startswith("--gpu="):
            val = a.split("=", 1)[1]; os.environ["CUDA_VISIBLE_DEVICES"] = val; return int(val)
    return None
_pin_gpu(sys.argv)

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "core"))
from common import (BACKBONES, EPOCHS, LR, MAX_LENGTH, TRAIN_SIZE,
                    evaluate_k0, load_base, load_k0, load_trivia,
                    get_batch_size, defrag, format_prompt)
from io_utils import load_json, save_json

SEED    = 15
BACKBONE = "sheared"

def sha_dir(d: Path) -> str:
    """Stable hash of all files in an adapter directory."""
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]

def train_one_gen(model, tok, corpus, cfg, gen_n, out_dir):
    """Fine-tune model on corpus for one generation. Returns model."""
    texts = [fmt["text"] if isinstance(fmt, dict) else fmt for fmt in corpus]
    ds = Dataset.from_dict({"text": texts})
    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=MAX_LENGTH, padding=False)
    tok_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    
    bs, accum = cfg["train_batch"], cfg["train_accum"]
    args = TrainingArguments(
        output_dir=str(out_dir / f"tmp_gen{gen_n}"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        learning_rate=LR,
        fp16=False, bf16=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    del trainer, tok_ds, ds
    gc.collect(); torch.cuda.empty_cache()
    return model

def gen_synthetic(model, tok, train_q, cfg):
    """Generate synthetic corpus from training questions."""
    from common import generate_synthetic
    return generate_synthetic(model, tok, train_q, cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    cfg = BACKBONES[BACKBONE]
    out_dir = ROOT / "outputs" / "sheared_g2"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "g2_result.json"
    if result_path.exists():
        print("G2 result already exists:", result_path)
        print(load_json(result_path, expect=dict))
        return

    print("=" * 60)
    print(f"G2 PILOT — Sheared-LLaMA seed={SEED} Gen0-3 (C-S arm)")
    print("=" * 60)

    # ── load data ──────────────────────────────────────────────────
    train_q, eval_q, eval_a = load_trivia()
    k0_q, k0_a, k0_meta     = load_k0(BACKBONE, eval_q, eval_a)
    print(f"|K0| = {len(k0_q)}")

    # Load Gen0 synthetic corpus (pre-computed by freeze_k0)
    syn0_path = ROOT / "outputs" / f"synthetic_gen0_{BACKBONE}.json"
    if syn0_path.exists():
        corpus = load_json(syn0_path, expect=list)
        print(f"Loaded Gen0 corpus from {syn0_path.name} ({len(corpus)} examples)")
    else:
        print("Gen0 corpus not found — generating fresh (slower)")
        corpus = None  # will generate after first eval

    # ── load model ─────────────────────────────────────────────────
    print(f"\nLoading {cfg['model']} on GPU {args.gpu}...")
    t_load = time.time()
    model, tok = load_base(cfg, quantized=True, device=args.gpu)
    lora_cfg = LoraConfig(r=cfg["rank"], lora_alpha=cfg["rank"] * 2,
                          target_modules=["q_proj", "v_proj"],
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    print(f"Model loaded in {time.time()-t_load:.1f}s")

    if corpus is None:
        corpus = gen_synthetic(model, tok, train_q, cfg)

    # ── Gen 0-2 shared prefix ──────────────────────────────────────
    timing = {}
    for gen in range(3):
        t0 = time.time()
        print(f"\n── Gen {gen} {'(eval+train)':>15} ──")
        k0_ret = evaluate_k0(model, tok, k0_q, k0_a, cfg)
        n_correct = sum(k0_ret)
        pct       = 100 * n_correct / len(k0_q)
        print(f"   Retention: {n_correct}/{len(k0_q)} = {pct:.1f}%")
        model = train_one_gen(model, tok, corpus, cfg, gen, out_dir)
        corpus = gen_synthetic(model, tok, train_q, cfg)
        timing[f"gen{gen}_s"] = round(time.time() - t0, 1)

    # Save Gen2 adapter (for pre-registration hash)
    ckpt2 = out_dir / "gen2_adapter"
    model.save_pretrained(str(ckpt2))
    ckpt2_hash = sha_dir(ckpt2)
    print(f"\nGen2 adapter saved → {ckpt2.name}  (sha={ckpt2_hash})")

    # ── Gen 3 — C-S arm (continued + synthetic): this is R3 ────────
    t0 = time.time()
    print(f"\n── Gen 3 (C-S arm — continued+synthetic) ──")
    k0_ret3 = evaluate_k0(model, tok, k0_q, k0_a, cfg)
    n3 = sum(k0_ret3)
    R3 = n3 / len(k0_q)
    print(f"   R3 = {n3}/{len(k0_q)} = {R3:.3f}  ({R3*100:.1f}%)")
    timing["gen3_eval_s"] = round(time.time() - t0, 1)

    # ── Gate decision ──────────────────────────────────────────────
    if R3 >= 0.40:
        decision = "PRIMARY"
        onset    = 3
        note     = "R3 >= 0.40: primary protocol, onset Gen3"
    elif R3 >= 0.20:
        decision = "FALLBACK"
        onset    = 2
        note     = "0.20 <= R3 < 0.40: fallback protocol, onset Gen2"
    else:
        decision = "STOP"
        onset    = None
        note     = "R3 < 0.20: K0 retention too low, do not proceed"

    result = {
        "backbone":       BACKBONE,
        "seed":           SEED,
        "k0_size":        len(k0_q),
        "R3":             round(R3, 6),
        "R3_pct":         round(R3 * 100, 2),
        "n3_correct":     n3,
        "gate_decision":  decision,
        "onset_gen":      onset,
        "note":           note,
        "gen2_adapter_sha": ckpt2_hash,
        "k0_indices_sha": k0_meta.get("indices_sha"),
        "timing_s":       timing,
    }
    save_json(result, result_path)

    print("\n" + "=" * 60)
    print(f"G2 RESULT: {decision}")
    print(f"  R3 = {R3*100:.1f}%  ({n3}/{len(k0_q)})")
    print(f"  Onset: Gen{onset}")
    print(f"  Gen2 adapter sha: {ckpt2_hash}")
    print(f"  {note}")
    print("=" * 60)
    print(f"\nNext: fill pre-registration template and commit before Phase-B of seed 1.")

    del model
    defrag()

if __name__ == "__main__":
    main()
