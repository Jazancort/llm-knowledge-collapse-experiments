"""Extract per-matrix effective rank from any LoRA config.

Loads base model, applies LoRA with given config, trains 1 gen on synthetic,
then reports effective rank for EACH matrix individually + aggregate.

Usage:
  uv run python scripts/extract_per_matrix_rank.py --rank 16 --target attention --seed 15
  uv run python scripts/extract_per_matrix_rank.py --rank 4 --target full_linear --seed 15
"""
import sys
import time
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TARGET_MODULES = {
    "attention": ["q_proj", "v_proj"],
    "full_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def compute_per_matrix_rank(model):
    """Return effective rank for each LoRA matrix individually."""
    results = {}
    params = dict(model.named_parameters())

    for name, param in params.items():
        if "lora_A" not in name or not param.requires_grad:
            continue
        b_name = name.replace("lora_A", "lora_B")
        b_param = params.get(b_name)
        if b_param is None:
            continue

        AB = b_param.data.float().cpu() @ param.data.float().cpu()
        svs = torch.linalg.svdvals(AB)
        svs_n = svs / (svs.sum() + 1e-10)
        eff_rank = torch.exp(-(svs_n * torch.log(svs_n + 1e-10)).sum()).item()

        # Extract module type from name (e.g. "model.layers.0.self_attn.q_proj.lora_A.default")
        parts = name.split(".")
        for p in parts:
            if p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                module_type = p
                break
        else:
            module_type = "unknown"

        # Extract layer number
        layer_num = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                layer_num = int(parts[i + 1])
                break

        key = f"layer{layer_num}.{module_type}"
        results[key] = eff_rank

    return results


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--target", choices=["attention", "full_linear"], required=True)
    parser.add_argument("--seed", type=int, default=15)
    args = parser.parse_args()

    target_modules = TARGET_MODULES[args.target]
    suffix = f"_full" if args.target == "full_linear" else ""
    output_dir = Path(__file__).parent.parent / "outputs" / f"g1_rank{args.rank}{suffix}_seed{args.seed}"

    print(f"Extracting per-matrix effective rank: r={args.rank}, target={args.target}")

    # Check if we already have a trained model's synthetic data (means training happened)
    syn_path = output_dir / "synthetic_gen1.json"
    if not syn_path.exists():
        print(f"ERROR: No synthetic_gen1.json found in {output_dir}. Run the ablation first.")
        return

    # Load synthetic from Gen0 for training
    with open(output_dir / "synthetic_gen0.json") as f:
        synthetic = json.load(f)

    # Load base model
    print("  Loading model...")
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

    # Apply LoRA and train 1 generation (same as ablation script)
    print(f"  Training LoRA r={args.rank} on {target_modules}...")
    torch.manual_seed(args.seed + 1)
    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=target_modules, task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    def tok_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=256, padding="max_length")
    train_ds = Dataset.from_dict({"text": synthetic})
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")

    t_train = time.time()
    trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(
            output_dir=str(output_dir / "tmp_extract"), num_train_epochs=2,
            per_device_train_batch_size=2, gradient_accumulation_steps=8,
            learning_rate=1e-5, bf16=True, logging_steps=9999,
            save_strategy="no", report_to="none", seed=args.seed + 1,
            gradient_checkpointing=len(target_modules) > 2,
            dataloader_num_workers=4,
        ),
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    print(f"  Training time: {time.time() - t_train:.1f}s")

    # Extract per-matrix effective rank
    print("  Extracting per-matrix effective ranks...")
    per_matrix = compute_per_matrix_rank(peft_model)

    # Aggregate by module type
    by_type = {}
    for key, eff in per_matrix.items():
        mod_type = key.split(".")[1]
        by_type.setdefault(mod_type, []).append(eff)

    print(f"\n  {'Module':<12} {'Mean Eff Rank':<15} {'Std':<8} {'Count':<6} {'Sum (28 layers)'}")
    print(f"  {'-'*55}")
    total_aggregate = 0
    for mod_type in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        if mod_type not in by_type:
            continue
        vals = by_type[mod_type]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        sum_val = sum(vals)
        total_aggregate += sum_val
        print(f"  {mod_type:<12} {mean_val:<15.3f} {std_val:<8.3f} {len(vals):<6} {sum_val:.1f}")

    print(f"\n  AGGREGATE EFFECTIVE RANK: {total_aggregate:.1f}")
    print(f"  (sum of all {len(per_matrix)} individual matrix effective ranks)")

    # Save results
    result = {
        "config": {"rank": args.rank, "target": args.target, "seed": args.seed},
        "trainable_params": trainable,
        "per_matrix": per_matrix,
        "by_module_type": {k: {"mean": np.mean(v), "std": np.std(v), "sum": sum(v), "count": len(v)} for k, v in by_type.items()},
        "aggregate_effective_rank": total_aggregate,
        "num_matrices": len(per_matrix),
    }
    out_path = output_dir / "per_matrix_rank.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(f"\n  Saved to: {out_path}")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
