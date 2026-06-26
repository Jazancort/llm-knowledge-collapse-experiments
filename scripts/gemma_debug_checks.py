"""Free checks for Gemma Sprint 2 debugging."""
import json, hashlib
from pathlib import Path

base = Path("outputs")
r16 = json.load(open(base / "gemma3_rank16_seed15/results.json"))
r256 = json.load(open(base / "gemma3_rank256_seed15/results.json"))

print("=== CHECK 1: SAME FACTS LOST? ===")
for gen_idx in [3, 5]:
    k0_r16 = r16[gen_idx].get("k0_results", [])
    k0_r256 = r256[gen_idx].get("k0_results", [])
    if not k0_r16 or not k0_r256:
        print(f"  Gen {gen_idx}: k0_results not available")
        continue
    lost_r16 = set(i for i, v in enumerate(k0_r16) if not v)
    lost_r256 = set(i for i, v in enumerate(k0_r256) if not v)
    overlap = lost_r16 & lost_r256
    union = lost_r16 | lost_r256
    jaccard = len(overlap) / len(union) if union else 0
    print(f"\n  Gen {gen_idx}:")
    print(f"    r=16  lost: {len(lost_r16)}/47")
    print(f"    r=256 lost: {len(lost_r256)}/47")
    print(f"    Same facts lost by both: {len(overlap)}")
    print(f"    Lost only by r=16: {len(lost_r16 - lost_r256)}")
    print(f"    Lost only by r=256: {len(lost_r256 - lost_r16)}")
    print(f"    Jaccard (lost facts): {jaccard:.3f}")

print("\n\n=== CHECK 2: COLLISION / DIFFERENT RUNS ===")
for gen in [0, 1, 3, 5]:
    h16 = hashlib.md5(open(base / f"gemma3_rank16_seed15/synthetic_gen{gen}.json", "rb").read()).hexdigest()[:12]
    h256 = hashlib.md5(open(base / f"gemma3_rank256_seed15/synthetic_gen{gen}.json", "rb").read()).hexdigest()[:12]
    same = "COLLISION!" if h16 == h256 else "OK (different)"
    print(f"  synthetic_gen{gen}: r16={h16}  r256={h256}  -> {same}")

print("\n  Gen0 synthetic (base model output, should be SAME for both):")
h16_0 = hashlib.md5(open(base / "gemma3_rank16_seed15/synthetic_gen0.json", "rb").read()).hexdigest()[:12]
h256_0 = hashlib.md5(open(base / "gemma3_rank256_seed15/synthetic_gen0.json", "rb").read()).hexdigest()[:12]
same0 = "SAME (expected)" if h16_0 == h256_0 else "DIFFERENT (unexpected!)"
print(f"    r16={h16_0}  r256={h256_0}  -> {same0}")
