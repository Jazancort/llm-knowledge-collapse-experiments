"""M1A: Infrastructure Validation Script.

Validates that all components work end-to-end WITHOUT training:
- Model loading (4-bit quantized)
- Inference with hidden states + attention extraction
- CKA computation (sanity check: model vs itself ≈ 1.0)
- Attention rollout
- ESI computation (self-comparison ≈ 0.0)
- Factual accuracy (exact match)
- Confidence metrics (log-prob, entropy)

Expected runtime: 30-60 min on RTX 3070.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from src.model import load_model, generate_with_states, get_hidden_and_attention
from src.explainability import linear_cka, compute_cka_by_blocks, extract_mean_pooled_states
from src.explainability.attention import attention_rollout, compute_esi
from src.evaluation import exact_match, batch_accuracy, token_log_probs, distinct_n
from src.utils import load_config, save_metadata

PROBE_SET = [
    {"question": "What is the capital of France?", "answers": ["Paris"]},
    {"question": "Who wrote Romeo and Juliet?", "answers": ["William Shakespeare", "Shakespeare"]},
    {"question": "What is the chemical symbol for gold?", "answers": ["Au"]},
    {"question": "What planet is closest to the Sun?", "answers": ["Mercury"]},
    {"question": "Who painted the Mona Lisa?", "answers": ["Leonardo da Vinci", "Da Vinci"]},
    {"question": "What is the largest ocean on Earth?", "answers": ["Pacific Ocean", "Pacific"]},
    {"question": "What year did World War II end?", "answers": ["1945"]},
    {"question": "What is the speed of light in km/s?", "answers": ["299792", "300000", "299792 km/s"]},
    {"question": "Who developed the theory of relativity?", "answers": ["Albert Einstein", "Einstein"]},
    {"question": "What is the smallest prime number?", "answers": ["2"]},
    {"question": "What is the capital of Japan?", "answers": ["Tokyo"]},
    {"question": "Who invented the telephone?", "answers": ["Alexander Graham Bell", "Bell"]},
    {"question": "What is the hardest natural substance?", "answers": ["Diamond"]},
    {"question": "What gas do plants absorb from the atmosphere?", "answers": ["Carbon dioxide", "CO2"]},
    {"question": "Who was the first person to walk on the Moon?", "answers": ["Neil Armstrong", "Armstrong"]},
    {"question": "What is the capital of Australia?", "answers": ["Canberra"]},
    {"question": "What is the largest mammal?", "answers": ["Blue whale"]},
    {"question": "In what year was the Berlin Wall torn down?", "answers": ["1989"]},
    {"question": "What element has atomic number 1?", "answers": ["Hydrogen"]},
    {"question": "Who wrote 1984?", "answers": ["George Orwell", "Orwell"]},
]


def format_prompt(question: str) -> str:
    return f"Answer the following question in 5 words or less.\nQuestion: {question}\nAnswer:"


def main():
    print("=" * 60)
    print("M1A: INFRASTRUCTURE VALIDATION")
    print("=" * 60)

    config_dir = Path(__file__).parent.parent / "configs"
    model_cfg = load_config(config_dir / "model.yaml")
    metrics_cfg = load_config(config_dir / "metrics.yaml")

    # --- Step 1: Load Model ---
    print("\n[1/6] Loading model...")
    model_name = model_cfg["pilot"]["name"]
    model, tokenizer = load_model(model_name, quantize_4bit=True)
    print(f"  Model: {model_name}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden dim: {model.config.hidden_size}")
    print(f"  Device: {next(model.parameters()).device}")

    # --- Step 2: Generation + Confidence ---
    print("\n[2/6] Running generation + confidence metrics...")
    predictions = []
    all_confidence = []

    for item in tqdm(PROBE_SET, desc="Generating"):
        prompt = format_prompt(item["question"])
        text, scores, inputs = generate_with_states(
            model, tokenizer, prompt, max_new_tokens=32, temperature=0.7
        )
        predictions.append(text.strip())

        generated_ids = tokenizer.encode(text, add_special_tokens=False)
        n_scores = min(len(scores), len(generated_ids))
        if n_scores > 0:
            conf = token_log_probs(scores[:n_scores], torch.tensor(generated_ids[:n_scores]))
            all_confidence.append(conf)

    # --- Step 3: Accuracy ---
    print("\n[3/6] Computing factual accuracy...")
    ground_truths = [item["answers"] for item in PROBE_SET]
    accuracy = batch_accuracy(predictions, ground_truths)

    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Sample predictions:")
    for i in range(min(5, len(predictions))):
        match = "✓" if exact_match(predictions[i], ground_truths[i]) else "✗"
        print(f"    {match} Q: {PROBE_SET[i]['question']}")
        print(f"      A: {predictions[i]} | GT: {ground_truths[i]}")

    if all_confidence:
        avg_log_prob = np.mean([c["avg_log_prob"] for c in all_confidence])
        avg_entropy = np.mean([c["avg_entropy"] for c in all_confidence])
        print(f"\n  Avg log-prob (confidence): {avg_log_prob:.4f}")
        print(f"  Avg entropy: {avg_entropy:.4f}")

    # --- Step 4: Hidden States + CKA ---
    print("\n[4/6] Extracting hidden states + computing CKA...")
    all_hidden_states = []
    all_attentions = []

    for item in tqdm(PROBE_SET, desc="Extracting states"):
        prompt = format_prompt(item["question"])
        hidden_states, attentions, _ = get_hidden_and_attention(model, tokenizer, prompt)
        all_hidden_states.append(hidden_states)
        all_attentions.append(attentions)

    pooled_states = extract_mean_pooled_states(all_hidden_states, len(PROBE_SET))
    n_layers = len(pooled_states)

    layer_blocks = {
        "early": [0, n_layers // 3],
        "middle": [n_layers // 3, 2 * n_layers // 3],
        "late": [2 * n_layers // 3, n_layers],
    }

    cka_self = compute_cka_by_blocks(pooled_states, pooled_states, layer_blocks)
    print(f"  CKA self-similarity (sanity check, expect ~1.0):")
    for block, val in cka_self.items():
        print(f"    {block}: {val:.4f}")

    # --- Step 5: Attention Rollout + ESI ---
    print("\n[5/6] Computing attention rollout + ESI...")
    rollouts = []
    for sample_attn in all_attentions:
        layers = [a.squeeze(0).numpy() for a in sample_attn]
        rollouts.append(attention_rollout(layers))

    esi_self_values = []
    for i in range(len(rollouts)):
        esi_result = compute_esi(rollouts[i], rollouts[i])
        esi_self_values.append(esi_result["esi"])

    avg_esi_self = np.mean(esi_self_values)
    print(f"  ESI self-comparison (sanity check, expect ~0.0): {avg_esi_self:.6f}")
    print(f"  Sample rollout shape: {rollouts[0].shape}")
    print(f"  Sample rollout top-5 positions: {np.argsort(rollouts[0])[-5:]}")

    # --- Step 6: Fluency (distinct-4) ---
    print("\n[6/6] Computing fluency metrics...")
    d4 = distinct_n(predictions, n=4)
    print(f"  Distinct-4: {d4:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    results = {
        "model": model_name,
        "num_layers": n_layers,
        "probe_size": len(PROBE_SET),
        "accuracy": accuracy,
        "avg_log_prob": float(avg_log_prob) if all_confidence else None,
        "avg_entropy": float(avg_entropy) if all_confidence else None,
        "distinct_4": d4,
        "cka_self": cka_self,
        "esi_self": float(avg_esi_self),
        "layer_blocks": layer_blocks,
    }

    for k, v in results.items():
        print(f"  {k}: {v}")

    # Sanity checks
    print("\n  SANITY CHECKS:")
    checks = [
        ("CKA self ≈ 1.0", all(v > 0.99 for v in cka_self.values())),
        ("ESI self ≈ 0.0", avg_esi_self < 0.01),
        ("Accuracy > 0%", accuracy > 0),
        ("Distinct-4 > 0", d4 > 0),
    ]
    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  ✓ All infrastructure checks passed. Ready for M1B.")
    else:
        print("\n  ✗ Some checks failed. Review before proceeding.")

    output_dir = Path(__file__).parent.parent / "outputs" / "m1a_validation"
    save_metadata(output_dir, **results)
    print(f"\n  Results saved to: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
