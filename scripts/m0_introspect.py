"""M0: Model Introspection.

Loads the pilot model (Qwen2.5-1.5B-Instruct, 4-bit) and reports:
- Architecture structure (named_modules)
- Exact path to decoder layers (for hooks)
- Number of layers, hidden dim, num heads
- VRAM usage after loading
- Candidate layers for monitoring (uniform sampling)
- Output format of decoder layers (tuple vs tensor)

Run: uv run python scripts/m0_introspect.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    print("=" * 60)
    print("M0: MODEL INTROSPECTION")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # --- Load ---
    print(f"\n[1/5] Loading {model_name} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # --- VRAM ---
    print(f"\n[2/5] VRAM usage:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Total GPU: {total:.2f} GB")
        print(f"  Free:      {total - reserved:.2f} GB")

    # --- Architecture ---
    print(f"\n[3/5] Model config:")
    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    print(f"  num_hidden_layers: {num_layers}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_attention_heads: {num_heads}")
    print(f"  num_key_value_heads: {num_kv_heads}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  max_position_embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")

    # --- Find decoder layers path ---
    print(f"\n[4/5] Module structure (finding decoder layers):")
    decoder_layer_path = None
    decoder_layer_type = None

    for name, module in model.named_modules():
        # Find the first indexed layer to determine path
        if ".layers.0" in name and name.endswith(".0"):
            # Go up one level to get the container
            parts = name.rsplit(".0", 1)
            decoder_layer_path = parts[0]
            decoder_layer_type = type(module).__name__
            break

    if decoder_layer_path:
        print(f"  Decoder layers path: {decoder_layer_path}")
        print(f"  Layer type: {decoder_layer_type}")
    else:
        # Fallback: search common patterns
        for name, module in model.named_modules():
            if name.endswith(".layers"):
                decoder_layer_path = name
                break
        print(f"  Decoder layers path (fallback): {decoder_layer_path}")

    # Print top-level structure
    print(f"\n  Top-level modules:")
    for name, module in model.named_children():
        print(f"    {name}: {type(module).__name__}")
        for child_name, child in module.named_children():
            print(f"      {name}.{child_name}: {type(child).__name__}")
            if hasattr(child, "layers"):
                print(f"        -> .layers: {len(child.layers)} decoder blocks")

    # Print single layer structure
    print(f"\n  Single decoder layer (layer 0) internal structure:")
    layer_0 = None
    for name, module in model.named_modules():
        if name == f"{decoder_layer_path}.0":
            layer_0 = module
            break

    if layer_0:
        for name, child in layer_0.named_children():
            print(f"    .{name}: {type(child).__name__}")
            for subname, subchild in child.named_children():
                print(f"      .{name}.{subname}: {type(subchild).__name__}")

    # --- Test forward pass and output format ---
    print(f"\n[5/5] Testing forward pass output format:")
    test_prompt = "What is the capital of France?"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    print(f"  Test prompt: '{test_prompt}'")
    print(f"  Token length: {prompt_len}")

    # Test with output_hidden_states to see format
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=False)

    print(f"  Output keys: {[k for k in outputs.keys()]}")
    print(f"  hidden_states: {len(outputs.hidden_states)} tensors")
    print(f"  hidden_states[0] shape: {outputs.hidden_states[0].shape}")
    print(f"  hidden_states[-1] shape: {outputs.hidden_states[-1].shape}")
    print(f"  logits shape: {outputs.logits.shape}")

    # Test hook output format
    hook_output = {}

    def test_hook(module, inp, out):
        if isinstance(out, tuple):
            hook_output["is_tuple"] = True
            hook_output["tuple_len"] = len(out)
            hook_output["tensor_shape"] = out[0].shape
            hook_output["tensor_dtype"] = out[0].dtype
        else:
            hook_output["is_tuple"] = False
            hook_output["tensor_shape"] = out.shape
            hook_output["tensor_dtype"] = out.dtype

    # Register on layer 0
    handle = layer_0.register_forward_hook(test_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    print(f"\n  Hook output format (layer 0):")
    for k, v in hook_output.items():
        print(f"    {k}: {v}")

    # --- Candidate layers ---
    step = max(1, num_layers // 8)
    candidates = list(range(1, num_layers - 1, step))
    if candidates[-1] != num_layers - 2:
        candidates.append(num_layers - 2)

    print(f"\n  Candidate monitoring layers ({len(candidates)} points):")
    print(f"    {candidates}")
    print(f"    (uniform sampling, step={step})")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("M0 SUMMARY")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_size}")
    print(f"  Decoder path: {decoder_layer_path}")
    print(f"  Hook output: {'tuple' if hook_output.get('is_tuple') else 'tensor'}, "
          f"shape {hook_output.get('tensor_shape')}, dtype {hook_output.get('tensor_dtype')}")
    print(f"  Monitor layers: {candidates}")
    print(f"  VRAM used: {allocated:.2f} GB / {total:.2f} GB")
    print(f"\n  Ready for M1A: {'YES' if decoder_layer_path and hook_output else 'NO'}")


if __name__ == "__main__":
    main()
