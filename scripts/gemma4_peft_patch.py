"""Patch PEFT to support Gemma4ClippableLinear modules.

Gemma 4 uses Gemma4ClippableLinear as a wrapper around Linear/Linear4bit.
PEFT doesn't recognize it. This patch unwraps the inner linear before LoRA injection.

Usage: import this module before calling get_peft_model() on a Gemma 4 model.
"""
import torch.nn as nn


def patch_gemma4_for_peft(model):
    """Replace Gemma4ClippableLinear wrappers with their inner linear modules.
    
    This allows PEFT/LoRA to target q_proj, v_proj etc. directly.
    Must be called AFTER model loading and BEFORE get_peft_model().
    """
    replacements = []
    
    for name, module in model.named_modules():
        if type(module).__name__ == "Gemma4ClippableLinear":
            # The inner linear is stored in module.linear
            if hasattr(module, "linear"):
                replacements.append((name, module.linear))
    
    for name, new_module in replacements:
        # Navigate to parent and set the attribute
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    print(f"  Patched {len(replacements)} Gemma4ClippableLinear -> Linear4bit")
    return model
