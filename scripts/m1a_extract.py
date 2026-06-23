"""M1A Step 1: Probe Extractor with forward hooks.

Extracts per-layer representations (global + factual) during inference.
Validates shapes, VRAM, and timing on 10 prompts before scaling to 200.

Run: uv run python scripts/m1a_extract.py
"""

import sys
import time
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
        self.buffer = {idx: {"global": [], "f1": [], "f3": [], "f5": []} for idx in monitor_layers}
        self.current_prompt_end_indices = []

    def _create_hook(self, layer_idx):
        def hook(module, input, output):
            # output is tuple for Qwen2DecoderLayer — tensor is output[0]
            tensor = output[0].detach()

            batch_size = tensor.size(0)
            for i in range(batch_size):
                end_idx = self.current_prompt_end_indices[i]

                # Reduce on GPU, then offload to CPU as float32
                glob = tensor[i, :end_idx + 1, :].mean(dim=0).cpu().float().numpy()
                f1 = tensor[i, end_idx, :].cpu().float().numpy()

                start_f3 = max(0, end_idx - 2)
                f3 = tensor[i, start_f3:end_idx + 1, :].mean(dim=0).cpu().float().numpy()

                start_f5 = max(0, end_idx - 4)
                f5 = tensor[i, start_f5:end_idx + 1, :].mean(dim=0).cpu().float().numpy()

                self.buffer[layer_idx]["global"].append(glob)
                self.buffer[layer_idx]["f1"].append(f1)
                self.buffer[layer_idx]["f3"].append(f3)
                self.buffer[layer_idx]["f5"].append(f5)

        return hook

    def register(self):
        for idx in self.monitor_layers:
            layer = self.model.model.layers[idx]
            self.hooks.append(layer.register_forward_hook(self._create_hook(idx)))

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_stacked(self, layer_idx: int, key: str) -> np.ndarray:
        """Stack buffer into (n_samples, hidden_dim) float32 array."""
        return np.stack(self.buffer[layer_idx][key])

    def reset_buffer(self):
        self.buffer = {idx: {"global": [], "f1": [], "f3": [], "f5": []} for idx in self.monitor_layers}


def compute_prompt_end_indices(tokenizer, prompts: list[str]) -> tuple[dict, list[int]]:
    """Tokenize prompts and return inputs + end indices (last real token before generation)."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    # End index = last non-pad token for each sequence
    attention_mask = inputs["attention_mask"]
    end_indices = (attention_mask.sum(dim=1) - 1).tolist()
    return inputs, end_indices


# --- Test prompts ---
TEST_PROMPTS = [
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
]


def main():
    print("=" * 60)
    print("M1A STEP 1: PROBE EXTRACTION VALIDATION")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    monitor_layers = [1, 4, 7, 10, 13, 16, 19, 22, 25, 26]
    batch_size = 4

    # Load
    print(f"\n[1/4] Loading {model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    model.eval()

    vram_post_load = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM after load: {vram_post_load:.2f} GB")

    # Setup extractor
    print(f"\n[2/4] Registering hooks on layers {monitor_layers}...")
    extractor = ProbeExtractor(model, monitor_layers)
    extractor.register()

    # Run inference in batches
    print(f"\n[3/4] Running {len(TEST_PROMPTS)} prompts (batch_size={batch_size})...")
    total_time = 0

    for batch_start in range(0, len(TEST_PROMPTS), batch_size):
        batch_prompts = TEST_PROMPTS[batch_start:batch_start + batch_size]
        inputs, end_indices = compute_prompt_end_indices(tokenizer, batch_prompts)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        extractor.current_prompt_end_indices = end_indices

        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            model(**inputs)

        torch.cuda.synchronize()
        elapsed = time.time() - t0
        total_time += elapsed

    vram_post_inference = torch.cuda.memory_allocated() / 1024**3
    vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    # Report
    print(f"\n[4/4] Results:")
    print(f"  Time total: {total_time:.2f}s ({total_time/len(TEST_PROMPTS)*1000:.0f}ms/prompt)")
    print(f"  VRAM after inference: {vram_post_inference:.2f} GB")
    print(f"  VRAM peak: {vram_peak:.2f} GB")

    print(f"\n  Buffer shapes (layer {monitor_layers[0]}):")
    for key in ["global", "f1", "f3", "f5"]:
        arr = extractor.get_stacked(monitor_layers[0], key)
        print(f"    {key}: {arr.shape}, dtype={arr.dtype}")

    print(f"\n  Buffer shapes (layer {monitor_layers[-1]}):")
    for key in ["global", "f1", "f3", "f5"]:
        arr = extractor.get_stacked(monitor_layers[-1], key)
        print(f"    {key}: {arr.shape}, dtype={arr.dtype}")

    print(f"\n  Prompt end indices: {extractor.current_prompt_end_indices}")

    # Sanity: check values are not NaN/Inf
    sample = extractor.get_stacked(monitor_layers[4], "f1")
    has_nan = np.isnan(sample).any()
    has_inf = np.isinf(sample).any()
    print(f"\n  NaN check (layer {monitor_layers[4]}, f1): {'FAIL' if has_nan else 'PASS'}")
    print(f"  Inf check (layer {monitor_layers[4]}, f1): {'FAIL' if has_inf else 'PASS'}")
    print(f"  Value range: [{sample.min():.4f}, {sample.max():.4f}]")

    # Cleanup
    extractor.clear_hooks()

    print("\n" + "=" * 60)
    print("M1A STEP 1: ", "PASSED" if not (has_nan or has_inf) else "FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
