#!/usr/bin/env python3
"""audit_k0_sheared.py -- Pre-launch K0 audit: emoji, overlap, echo-FP."""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "core"))
from common import DATASET_SEED, TRAIN_SIZE, EVAL_SIZE

def normalize(text):
    text = text.lower().strip()
    nums = {"zero":"0","one":"1","two":"2","three":"3","four":"4",
            "five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
    for w, d in nums.items():
        text = re.sub(rf"\b{w}\b", d, text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def has_emoji(s):
    import unicodedata
    return any(unicodedata.category(c) in ("So","Sm","Sk") or
               0x1F000 <= ord(c) <= 0x1FFFF or 0x2600 <= ord(c) <= 0x27FF
               for c in s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="sheared")
    ap.add_argument("--k0", default=None)
    args = ap.parse_args()

    k0_path = Path(args.k0) if args.k0 else ROOT/"outputs"/f"k0_canonical_{args.backbone}.json"
    if not k0_path.exists():
        sys.exit(f"K0 not found: {k0_path}")
    k0 = json.loads(k0_path.read_text())
    k0_q = k0["k0_questions"]; k0_idx = k0["k0_indices"]
    print(f"|K0|={k0['k0_size']}  model={k0.get('model','?')}\n")

    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa","rc.nocontext",split="train",trust_remote_code=True)
    ds = ds.shuffle(seed=DATASET_SEED)
    train_q_set = {ds[i]["question"].strip().lower() for i in range(TRAIN_SIZE)}
    eval_items = [{"q": ds[i]["question"],
                   "aliases": ds[i]["answer"]["aliases"]+[ds[i]["answer"]["value"]]}
                  for i in range(TRAIN_SIZE, TRAIN_SIZE+EVAL_SIZE)]

    emoji_items, overlap_items, echo_items = [], [], []

    for idx in k0_idx:
        item = eval_items[idx]
        q, aliases = item["q"], item["aliases"]

        # A. emoji
        ea = [a for a in aliases if has_emoji(a)]
        if ea: emoji_items.append({"idx":idx,"q":q,"emoji_aliases":ea})

        # B. overlap
        if q.strip().lower() in train_q_set:
            overlap_items.append({"idx":idx,"q":q})

        # C. echo FP
        q_n = normalize(q)
        ea2 = [a for a in aliases if (an:=normalize(a)) and an in q_n]
        if ea2: echo_items.append({"idx":idx,"q":q,"echo_aliases":ea2})

    print(f"A. Emoji-alias items  : {len(emoji_items)}")
    for e in emoji_items: print(f"   [{e['idx']}] {e['q'][:60]}  =>  {e['emoji_aliases']}")
    print(f"\nB. Train overlap      : {len(overlap_items)}")
    for o in overlap_items: print(f"   [{o['idx']}] {o['q'][:80]}")
    print(f"\nC. Echo-FP risk items : {len(echo_items)}")
    for e in echo_items[:10]: print(f"   [{e['idx']}] {e['q'][:60]}  =>  {e['echo_aliases'][:2]}")
    if len(echo_items) > 10: print(f"   ... ({len(echo_items)-10} more)")

    concern = len(set([e['idx'] for e in emoji_items]+
                      [o['idx'] for o in overlap_items]+
                      [e['idx'] for e in echo_items]))
    print(f"\nTotal items ≥1 concern: {concern}/{k0['k0_size']}")
    print("Emoji check:", "PASS (0 items)" if not emoji_items else f"FAIL ({len(emoji_items)} items)")

    out = ROOT/"outputs"/f"audit_k0_{args.backbone}.json"
    out.write_text(json.dumps({"emoji":emoji_items,"overlap":overlap_items,
                               "echo":echo_items,"k0_size":k0["k0_size"]},
                              indent=2, ensure_ascii=False))
    print(f"\nReport -> {out}")

if __name__ == "__main__":
    main()
