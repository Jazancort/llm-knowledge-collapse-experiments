"""M1A Step 2: CKA Identity & Sensitivity Tests.

Validates:
1. CKA(M,M) ≈ 1.0 (identity - same model, same prompts, two passes)
2. CKA(M, M+ε) decreases monotonically with perturbation magnitude
3. CKA-Factual vs CKA-Global sensitivity comparison

Run: uv run python scripts/m1a_cka_tests.py
"""

import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ProbeExtractor:
    def __init__(self, model, monitor_layers: list[int]):
        self.model = model
        self.monitor_layers = monitor_layers
        self.hooks = []
        self.buffer = {idx: {"global": [], "f1": []} for idx in monitor_layers}
        self.current_prompt_end_indices = []

    def _create_hook(self, layer_idx):
        def hook(module, input, output):
            tensor = output[0].detach()
            batch_size = tensor.size(0)
            for i in range(batch_size):
                end_idx = self.current_prompt_end_indices[i]
                glob = tensor[i, :end_idx + 1, :].mean(dim=0).cpu().float().numpy()
                f1 = tensor[i, end_idx, :].cpu().float().numpy()
                self.buffer[layer_idx]["global"].append(glob)
                self.buffer[layer_idx]["f1"].append(f1)
        return hook

    def register(self):
        for idx in self.monitor_layers:
            layer = self.model.model.layers[idx]
            self.hooks.append(layer.register_forward_hook(self._create_hook(idx)))

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_stacked(self, layer_idx, key):
        return np.stack(self.buffer[layer_idx][key])

    def reset_buffer(self):
        self.buffer = {idx: {"global": [], "f1": []} for idx in self.monitor_layers}


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA. Expects float32 arrays of shape (n_samples, features)."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XtX = X @ X.T
    YtY = Y @ Y.T
    XtY = X @ Y.T
    num = np.trace(XtX @ YtY)
    denom = np.linalg.norm(XtX, ord='fro') * np.linalg.norm(YtY, ord='fro')
    return float(num / (denom + 1e-10))


def run_extraction(model, tokenizer, extractor, prompts, batch_size=4):
    """Run forward pass and fill extractor buffer."""
    extractor.reset_buffer()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        end_indices = (inputs["attention_mask"].sum(dim=1) - 1).tolist()
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        extractor.current_prompt_end_indices = end_indices
        with torch.no_grad():
            model(**inputs)


PROMPTS = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "What planet is closest to the Sun?",
    "Who painted the Mona Lisa?",
    "What is the largest ocean on Earth?",
    "What year did World War II end?",
    "Who developed the theory of relativity?",
    "What is the smallest prime number?",
    "What is the capital of Japan?",
    "Who invented the telephone?",
    "What is the hardest natural substance?",
    "What gas do plants absorb?",
    "Who was the first person on the Moon?",
    "What is the capital of Australia?",
    "What is the largest mammal?",
    "In what year was the Berlin Wall torn down?",
    "What element has atomic number 1?",
    "Who wrote 1984?",
    "What is the boiling point of water in Celsius?",
]


def main():
    print("=" * 60)
    print("M1A STEP 2: CKA IDENTITY & SENSITIVITY")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    monitor_layers = [1, 7, 13, 19, 25]  # subset for speed

    # Load
    print(f"\n[1/4] Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # --- TEST 1: Identity CKA(M, M) ---
    print(f"\n[2/4] Test 1: Identity CKA(M, M)...")
    extractor = ProbeExtractor(model, monitor_layers)
    extractor.register()

    # Pass 1
    run_extraction(model, tokenizer, extractor, PROMPTS)
    pass1 = {idx: {"global": extractor.get_stacked(idx, "global"),
                   "f1": extractor.get_stacked(idx, "f1")} for idx in monitor_layers}

    # Pass 2 (identical)
    run_extraction(model, tokenizer, extractor, PROMPTS)
    pass2 = {idx: {"global": extractor.get_stacked(idx, "global"),
                   "f1": extractor.get_stacked(idx, "f1")} for idx in monitor_layers}

    print(f"\n  {'Layer':<8} {'CKA-Global':<14} {'CKA-Factual':<14}")
    print(f"  {'-'*36}")
    identity_ok = True
    for idx in monitor_layers:
        cka_g = linear_cka(pass1[idx]["global"], pass2[idx]["global"])
        cka_f = linear_cka(pass1[idx]["f1"], pass2[idx]["f1"])
        status = "OK" if cka_g > 0.999 and cka_f > 0.999 else "WARN"
        if cka_g < 0.99 or cka_f < 0.99:
            identity_ok = False
            status = "FAIL"
        print(f"  {idx:<8} {cka_g:<14.6f} {cka_f:<14.6f} {status}")

    # --- TEST 2: Sensitivity CKA(M, M+ε) ---
    print(f"\n[3/4] Test 2: Sensitivity to perturbation...")
    epsilons = [1e-3, 1e-2, 5e-2, 1e-1, 3e-1]
    # Use pass1 as reference, perturb the stored representations
    ref_layer = monitor_layers[2]  # middle layer
    ref_global = pass1[ref_layer]["global"]
    ref_f1 = pass1[ref_layer]["f1"]

    print(f"\n  Perturbing representations (layer {ref_layer}):")
    print(f"  {'Epsilon':<12} {'CKA-Global':<14} {'CKA-Factual':<14}")
    print(f"  {'-'*40}")

    prev_cka_g = 1.0
    prev_cka_f = 1.0
    monotonic = True

    for eps in epsilons:
        noise = np.random.randn(*ref_global.shape).astype(np.float32) * eps
        perturbed_global = ref_global + noise * np.std(ref_global)
        noise_f = np.random.randn(*ref_f1.shape).astype(np.float32) * eps
        perturbed_f1 = ref_f1 + noise_f * np.std(ref_f1)

        cka_g = linear_cka(ref_global, perturbed_global)
        cka_f = linear_cka(ref_f1, perturbed_f1)

        if cka_g > prev_cka_g or cka_f > prev_cka_f:
            monotonic = False
        prev_cka_g = cka_g
        prev_cka_f = cka_f

        print(f"  {eps:<12.0e} {cka_g:<14.6f} {cka_f:<14.6f}")

    # --- TEST 3: Sensitivity via actual weight perturbation ---
    print(f"\n[4/4] Test 3: Weight perturbation (layer {ref_layer})...")
    # Perturb actual model weights and re-extract
    target_layer = model.model.layers[ref_layer]

    weight_results = []
    for eps in [1e-2, 5e-2, 1e-1]:
        # Save original weights
        original_params = {}
        for name, param in target_layer.named_parameters():
            if param.requires_grad or "weight" in name:
                try:
                    original_params[name] = param.data.clone()
                except Exception:
                    continue

        # Perturb
        perturbed_count = 0
        for name, param in target_layer.named_parameters():
            if name in original_params and param.data.dtype in (torch.float16, torch.bfloat16, torch.float32):
                noise = torch.randn_like(param.data) * eps
                param.data.add_(noise)
                perturbed_count += 1

        # Re-extract
        run_extraction(model, tokenizer, extractor, PROMPTS)
        perturbed_pass = {idx: {"global": extractor.get_stacked(idx, "global"),
                                "f1": extractor.get_stacked(idx, "f1")} for idx in monitor_layers}

        cka_g = linear_cka(pass1[ref_layer]["global"], perturbed_pass[ref_layer]["global"])
        cka_f = linear_cka(pass1[ref_layer]["f1"], perturbed_pass[ref_layer]["f1"])
        weight_results.append((eps, cka_g, cka_f, perturbed_count))

        # Restore
        for name, param in target_layer.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])

    print(f"  {'Epsilon':<12} {'CKA-Global':<14} {'CKA-Factual':<14} {'Params perturbed'}")
    print(f"  {'-'*56}")
    for eps, cka_g, cka_f, count in weight_results:
        print(f"  {eps:<12.0e} {cka_g:<14.6f} {cka_f:<14.6f} {count}")

    # --- Summary ---
    extractor.clear_hooks()
    print("\n" + "=" * 60)
    print("M1A STEP 2 SUMMARY")
    print("=" * 60)
    print(f"  Identity test:    {'PASS' if identity_ok else 'FAIL'}")
    print(f"  Monotonic curve:  {'PASS' if monotonic else 'FAIL'}")
    print(f"  Weight perturb:   {len(weight_results)} levels tested")
    print(f"  CKA-Factual more sensitive than Global: checking...")

    # Compare sensitivity
    if weight_results:
        _, g_last, f_last, _ = weight_results[-1]
        factual_more_sensitive = f_last < g_last
        print(f"    Global CKA at eps=1e-3: {g_last:.6f}")
        print(f"    Factual CKA at eps=1e-3: {f_last:.6f}")
        print(f"    Factual more sensitive: {factual_more_sensitive}")

    all_pass = identity_ok and monotonic
    print(f"\n  Overall: {'PASSED — Ready for M1A Step 3' if all_pass else 'ISSUES DETECTED'}")


if __name__ == "__main__":
    main()
