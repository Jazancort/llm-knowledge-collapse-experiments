"""Audit LoRA norm metrics: compare compute_lora_drift vs compute_lora_norm on same model.

Loads a QLoRA model, trains 1 step, prints per-module breakdown of both metrics.
Zero-cost: runs on any GPU in ~2 minutes.

Run: uv run python scripts/audit_lora_norm.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    # Load model with QLoRA
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                                  device_map="auto", torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    # Quick 1-step train to get non-zero B
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset
    dummy = Dataset.from_dict({"text": ["Hello world this is a test"] * 16})
    dummy = dummy.map(lambda x: tok(x["text"], truncation=True, max_length=64, padding="max_length"), batched=True, remove_columns=["text"])
    dummy.set_format("torch")
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir="/tmp/audit", num_train_epochs=1, per_device_train_batch_size=16,
        learning_rate=1e-5, save_strategy="no", report_to="none", logging_steps=999),
        train_dataset=dummy, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()
    model.eval()
    del trainer

    # === Audit per module ===
    print("\n" + "=" * 100)
    print("LORA NORM AUDIT — Per Module")
    print("=" * 100)
    print(f"\n{'Module':<45} {'||A||':>8} {'||B||':>8} {'||B@A||':>8} {'scaled':>8} {'||W0||':>10} {'rel_delta':>10}")
    print("-" * 100)

    scaling = lora_config.lora_alpha / lora_config.r  # 32/16 = 2.0

    # Method 1: ||B@A|| per pair (what compute_lora_drift does)
    ba_norms_sq = []
    # Method 2: ||A|| + ||B|| raw (what compute_lora_norm does)
    raw_norms_sq = []

    params = dict(model.named_parameters())
    base_params = {n: p for n, p in model.named_parameters() if not any(x in n for x in ["lora_"])}

    for name, param in model.named_parameters():
        if "lora_A" in name and param.requires_grad:
            b_name = name.replace("lora_A", "lora_B")
            if b_name not in params:
                continue
            A = param.data.float().cpu()
            B = params[b_name].data.float().cpu()
            BA = B @ A
            scaled_BA = scaling * BA

            norm_A = A.norm().item()
            norm_B = B.norm().item()
            norm_BA = BA.norm().item()
            norm_scaled = scaled_BA.norm().item()

            # Find base weight for relative comparison
            # Module path: e.g. base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
            # Base:         base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
            base_name = name.replace("lora_A.default.weight", "base_layer.weight")
            W0_norm = 0.0
            if base_name in params:
                W0_norm = params[base_name].data.float().cpu().norm().item()
            rel_delta = norm_scaled / W0_norm if W0_norm > 0 else float('nan')

            # Short module name
            short = name.replace("base_model.model.model.", "").replace(".lora_A.default.weight", "")

            print(f"{short:<45} {norm_A:>8.4f} {norm_B:>8.4f} {norm_BA:>8.4f} {norm_scaled:>8.4f} {W0_norm:>10.4f} {rel_delta:>10.6f}")

            ba_norms_sq.append(norm_scaled ** 2)
            raw_norms_sq.append(norm_A ** 2 + norm_B ** 2)

    print("-" * 100)
    method1 = sum(ba_norms_sq) ** 0.5
    method2 = sum(raw_norms_sq) ** 0.5
    print(f"\n  Method 1 — ||scaling * B @ A|| aggregated (compute_lora_drift style): {method1:.4f}")
    print(f"  Method 2 — ||A|| + ||B|| raw aggregated (compute_lora_norm style):     {method2:.4f}")
    print(f"\n  Scaling factor (alpha/r): {scaling}")
    print(f"  Ratio method2/method1: {method2/method1:.1f}x")
    print(f"\n  NOTE: The fft_lr_sweep.py reports method1 WITHOUT scaling (just ||B@A||).")
    print(f"        With scaling=2.0, the effective perturbation would be {method1:.4f}")
    print(f"        Without scaling (raw B@A): {sum(n/4 for n in ba_norms_sq)**0.5:.4f}")

    # Compare with FFT drift
    print(f"\n  FFT LR=1e-6 drift: ~0.39")
    print(f"  QLoRA ||B@A|| (no scaling): {sum(n/4 for n in ba_norms_sq)**0.5:.4f}")
    print(f"  QLoRA ||scaling*B@A||: {method1:.4f}")
    print(f"\n  CONCLUSION: The 'lora_norm=0.42' from the sweep measures ||B@A|| without scaling.")
    print(f"             The 'lora_norm=17.38' from replicate measures ||all lora params|| raw.")
    print(f"             For perturbation-matching with FFT, use ||B@A|| ≈ 0.42 ≈ FFT drift 0.39.")


if __name__ == "__main__":
    main()
