"""Recursive fine-tuning with a switchable intervention regime.

The ETP scripts apply the C5 intervention uniformly from Gen1. This script adds
the one parameter that paper never had: WHEN the intervention starts.

  --switch-gen 0    C5 active from Gen1        (condition B, prevention)
  --switch-gen N    C1 for Gen1..N-1, then C5  (conditions W2/W5/W8/W11)
  --switch-gen inf  never intervene            (condition A, baseline)

The intervention is C5: random downsampling of the synthetic corpus. Dose is
RELATIVE (remove X% of the tokens generated in that generation) rather than the
absolute TARGET_TOKENS=54500 of the ETP script — see §5.4 and the note in
docs/EXPERIMENT_PLAN.md, which was revised after measuring the real corpora.

Per generation it records: retention, the full per-item K0 vector, the corpus
actually fed in, diversity metrics, the LoRA spectrum, and whether the
intervention was active. `intervention_active` makes each file self-explanatory
without consulting the invocation — one of the two provenance gaps found in the
ETP outputs (docs/BASELINE_DATA.md §3).

CRASH SAFETY (Athena is shared with LM Studio; OOM and reboots happen):
  - every write is atomic (temp file + fsync + os.replace)
  - re-running the same command resumes at generation len(results)+1
  - resume validates that the corpus it is about to train on is intact and the
    right size, and that the K0 hash still matches; it refuses to silently
    train on a truncated corpus
  - a status.json heartbeat is written at every phase boundary, so progress is
    readable from another shell without attaching to tmux
  - RUNNING.lock refuses two concurrent writers in the same output directory

Run on Athena:
  tmux new -s w5
  uv run python scripts/run_window.py --backbone qwen --switch-gen inf --seed 15
  uv run python scripts/run_window.py --backbone qwen --switch-gen 6   --seed 15
  uv run python scripts/run_window.py --backbone gemma --switch-gen 6  --seed 15 --dose 0.20

  detach: Ctrl+B then D      reattach: tmux attach -t w5
"""
import argparse, gc, os, socket, sys, time
from pathlib import Path


def _pin_visible_gpu(argv):
    """Restrict this process to ONE GPU, before torch is imported.

    This must run first, and it is not merely an optimisation:

    HuggingFace TrainingArguments inspects torch.cuda.device_count() and, if it
    sees more than one device, silently enables DataParallel. Measured on Athena
    with both cards visible:

        n_gpu = 2
        train_batch_size = per_device_train_batch_size * 2   -> 8, not 4
        args.device = cuda:0                                 -> ignores our pick

    So the effective batch became 32 instead of 16, both parallel trajectories
    piled onto GPU 0, and device_map={"": 1} was overridden. Every guarantee in
    §5.6 (fixed batch geometry) and §6.1 (one trajectory per card) was silently
    void. Setting CUDA_VISIBLE_DEVICES makes device_count() == 1, which is the
    only reliable way to keep the Trainer single-device.

    Parsed by hand because argparse runs long after `import torch` would have
    latched the device list. Accepts "--gpu N" and "--gpu=N".
    """
    val = None
    for i, a in enumerate(argv):
        if a == "--gpu" and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith("--gpu="):
            val = a.split("=", 1)[1]
    if val is None:
        return None                     # no --gpu: pick_gpu() chooses, see below
    if not val.lstrip("-").isdigit():
        raise SystemExit(f"--gpu must be an integer, got {val!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = val
    # CUDA_VISIBLE_DEVICES renumbers devices: the chosen card is now cuda:0 and
    # it is the ONLY one. Callers must therefore use index 0 downstream, not the
    # physical index, which is what --gpu means to the user.
    return int(val)


PHYSICAL_GPU = _pin_visible_gpu(sys.argv)

import numpy as np                      # noqa: E402  (must follow the pin)
import torch                            # noqa: E402
from transformers import (TrainingArguments, Trainer,  # noqa: E402
                          DataCollatorForLanguageModeling, BitsAndBytesConfig,
                          AutoModelForCausalLM, AutoTokenizer)
from peft import LoraConfig, get_peft_model    # noqa: E402
from datasets import Dataset                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "core"))

from common import (  # noqa: E402
    BACKBONES, EPOCHS, GENERATIONS, LR, MAX_LENGTH, TRAIN_SIZE,
    clean_stale_tmp, compute_diversity, compute_lora_spectrum, defrag,
    evaluate_k0, generate_synthetic, get_batch_size, load_json, load_k0,
    load_trivia, pick_gpu, resume_point, save_json,
)


def downsample_relative(texts, dose, rng_seed):
    """C5 with a relative dose: drop whole examples until `dose` of the
    whitespace-token mass of THIS generation has been removed.

    Whole examples, not truncation, so surviving text is untouched — the
    intervention changes corpus size, never the token distribution inside a kept
    example. Same mechanism as c5_replicate_masks.downsample, except the target
    is recomputed per generation instead of being a fixed constant.
    """
    total = sum(len(t.split()) for t in texts)
    target = total * (1.0 - dose)
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(len(texts))
    kept, acc = [], 0
    for i in order:
        n = len(texts[i].split())
        if acc + n > target:
            continue          # skip and keep filling; do not break early
        kept.append(texts[i])
        acc += n
    if not kept:              # pathological guard, as in the ETP script
        kept = list(texts[:100])
        acc = sum(len(t.split()) for t in kept)
    return kept, {"tokens_before": total, "tokens_after": acc,
                  "removed_frac": round(1 - acc / total, 4) if total else None,
                  "examples_before": len(texts), "examples_after": len(kept)}


def _pid_alive(pid):
    """True if the pid is running. Used only to decide whether a lock is stale.

    Fails safe: anything ambiguous returns True, i.e. do NOT reclaim the lock.
    A false "dead" would let two processes write the same results.json, which is
    far worse than a false "alive" (the user deletes the lock by hand).

    On POSIX this is exact. On Windows os.kill raises a generic OSError for a
    missing pid, indistinguishable from other failures, so this returns True and
    stale locks must be removed manually. Athena is Linux; this only affects
    local development.
    """
    if not isinstance(pid, int):
        return True
    try:
        os.kill(pid, 0)      # signal 0: existence check, no signal delivered
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by another user
    except OSError:
        return True          # Windows, or anything unexpected
    return True


def condition_name(switch_gen, dose):
    if switch_gen == float("inf"):
        return "A_baseline"
    if switch_gen == 0:
        return f"B_prevent_d{int(dose*100)}"
    return f"W{switch_gen}_d{int(dose*100)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="qwen")
    ap.add_argument("--switch-gen", required=True,
                    help="0 = intervene from Gen1; N = switch at Gen N; inf = never")
    ap.add_argument("--dose", type=float, default=0.05,
                    help="fraction of this generation's tokens to remove (default 0.05)")
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--generations", type=int, default=GENERATIONS)
    ap.add_argument("--save-adapters", default="switch",
                    choices=["none", "switch", "all"],
                    help="switch = Gen1, the switch neighbourhood, and the last gen (§5.2)")
    ap.add_argument("--keep-corpora", default="recent", choices=["recent", "all"],
                    help="recent = keep only the last 2 corpora on disk (~50 MB each)")
    ap.add_argument("--gpu", type=int, default=None,
                    help="force a CUDA device; default picks the one with most free memory")
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    ap.add_argument("--rank", type=int, default=None,
                    help="override backbone rank (default: from common.BACKBONES)")
    ap.add_argument("--lr", type=float, default=None,
                    help="override learning rate (default: common.LR)")
    args = ap.parse_args()

    switch = float("inf") if args.switch_gen.lower() in ("inf", "none", "never") \
        else int(args.switch_gen)
    if switch != float("inf") and not (0 <= switch <= args.generations):
        raise SystemExit(f"--switch-gen must be 0..{args.generations} or inf")
    if not 0.0 < args.dose < 1.0:
        raise SystemExit("--dose must be in (0, 1)")

    cfg = BACKBONES[args.backbone]
    # CLI overrides — allow varying rank and LR without editing common.py
    if args.rank is not None:
        cfg = dict(cfg)  # shallow copy: never mutate the module-level dict
        cfg["rank"] = args.rank
    _lr = args.lr if args.lr is not None else LR
    cond = condition_name(switch, args.dose)
    name = f"{args.backbone}_{cond}_seed{args.seed}{args.tag}"
    out_dir = ROOT / "outputs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.json"
    status_path = out_dir / "status.json"

    stale = clean_stale_tmp(out_dir)
    if stale:
        print(f"  cleaned {stale} stale .tmp file(s) from a previous kill")

    # --- single-writer lock: two runs in the same directory would interleave
    # their results.json writes and silently produce a mixed trajectory.
    lock = out_dir / "RUNNING.lock"
    if lock.exists():
        info = load_json(lock, expect=dict)
        if info.get("host") == socket.gethostname() and not _pid_alive(info.get("pid")):
            # Same machine, process gone: an OOM kill or a reboot. Reclaiming is
            # safe, and refusing here would block resume exactly when it matters.
            print(f"  stale lock from dead pid {info.get('pid')} "
                  f"(started {info.get('started')}) — reclaiming")
            lock.unlink(missing_ok=True)
        else:
            print(f"  WARNING: {lock.name} exists (pid {info.get('pid')} on "
                  f"{info.get('host')}, started {info.get('started')}).")
            print(f"  If that run is dead:  rm {lock}")
            raise SystemExit("  Refusing to run two writers in the same output dir.")
    save_json({"pid": os.getpid(), "host": socket.gethostname(),
               "started": time.strftime("%Y-%m-%d %H:%M:%S"),
               "argv": sys.argv[1:]}, lock)

    try:
        run(args, cfg, cond, name, out_dir, result_path, status_path, switch, _lr)
    finally:
        lock.unlink(missing_ok=True)


def run(args, cfg, cond, name, out_dir, result_path, status_path, switch, _lr):
    if PHYSICAL_GPU is not None:
        # CUDA_VISIBLE_DEVICES already narrowed the world to one card, which is
        # now cuda:0 regardless of its physical index.
        gpu = pick_gpu(prefer=0)
        print(f"  pinned to physical GPU {PHYSICAL_GPU} "
              f"(CUDA_VISIBLE_DEVICES={PHYSICAL_GPU}, visible as cuda:0)")
    else:
        gpu = pick_gpu()

    # Non-negotiable: with >1 visible device the Trainer enables DataParallel and
    # doubles the effective batch behind our back (see _pin_visible_gpu). Fail
    # here rather than produce a trajectory that silently used batch 32.
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            f"{torch.cuda.device_count()} GPUs visible; this run requires exactly 1.\n"
            f"  HuggingFace Trainer would enable DataParallel and double the "
            f"effective batch, breaking the fixed geometry (§5.6).\n"
            f"  Pass --gpu N, or export CUDA_VISIBLE_DEVICES=N.")

    # --- data and the frozen K0
    train_q, eval_q, eval_a = load_trivia()
    k0_q, k0_a, k0_meta = load_k0(args.backbone, eval_q, eval_a)
    K0 = len(k0_q)

    syn0 = ROOT / "outputs" / f"synthetic_gen0_{args.backbone}.json"
    if not syn0.exists():
        raise SystemExit(f"Missing Gen0 corpus: {syn0.name}\n"
                         f"Run:  uv run python scripts/freeze_k0.py --backbone {args.backbone}")

    # --- resume
    results = load_json(result_path, expect=list) if result_path.exists() else []
    start_gen, done = resume_point(results, args.generations)
    if start_gen > args.generations:
        print(f"[{name}] already complete ({done} generations): "
              f"{' '.join(str(r['retention']) for r in results)}")
        return

    # Resume integrity: the recorded K0 must be the K0 we just loaded, else the
    # trajectory would silently mix two evaluation bases (the exact failure mode
    # that made the ETP series non-interchangeable — §5.1).
    if results:
        prev_sha = (load_json(out_dir / "meta.json", expect=dict).get("k0_indices_sha")
                    if (out_dir / "meta.json").exists() else None)
        if prev_sha and prev_sha != k0_meta["indices_sha"]:
            raise SystemExit(
                f"K0 changed since this trajectory started.\n"
                f"  recorded {prev_sha}, current {k0_meta['indices_sha']}\n"
                f"  Do not mix bases. Re-run from scratch or restore the old K0.")
        if results[-1]["k0_size"] != K0:
            raise SystemExit(f"K0 size changed: {results[-1]['k0_size']} -> {K0}.")

    def adapter_wanted(gen):
        if args.save_adapters == "all":
            return True
        if args.save_adapters == "none":
            return False
        keep = {1, args.generations}
        if switch != float("inf"):
            keep |= {max(1, switch - 1), switch, min(args.generations, switch + 1)}
        return gen in keep

    print("=" * 66)
    print(f"  {name}")
    print("=" * 66)
    print(f"  model        : {cfg['model']}  (QLoRA r={cfg['rank']})")
    print(f"  K0           : {K0} items  (1 item = {100/K0:.2f}pp)  "
          f"sha={k0_meta['indices_sha']}")
    print(f"  switch       : {'never' if switch == float('inf') else f'Gen{switch}'}"
          f"   dose = {args.dose:.0%} relative")
    print(f"  generations  : {start_gen}..{args.generations}"
          + (f"   (resuming, {len(results)} done)" if results else ""))
    print(f"  adapters     : {args.save_adapters}   corpora: {args.keep_corpora}")

    # --- the corpus this generation trains on
    syn_path = out_dir / f"syn_gen{start_gen-1}.json"
    if not syn_path.exists():
        if start_gen != 1:
            raise SystemExit(
                f"Cannot resume: missing {syn_path.name}.\n"
                f"  Gen{start_gen} needs the corpus produced by Gen{start_gen-1}.\n"
                f"  Either restore it, or truncate results.json to "
                f"{start_gen-2} entries and re-run to regenerate.")
        syn_path = syn0

    t_run = time.time()
    for gen in range(start_gen, args.generations + 1):
        prev = load_json(syn_path, expect=list)
        if len(prev) < 100:
            raise SystemExit(
                f"Refusing to train on {syn_path.name}: only {len(prev)} examples.\n"
                f"  Expected ~{TRAIN_SIZE} (or ~{int(TRAIN_SIZE*(1-args.dose))} "
                f"after C5). Likely a truncated write — delete it and re-run.")
        active = switch != float("inf") and gen >= max(1, switch)
        t0 = time.time()
        print(f"\n  [Gen {gen}/{args.generations}]  in={len(prev)} ex  "
              f"regime={'C5' if active else 'C1'}")

        save_json({"gen": gen, "of": args.generations, "phase": "train",
                   "condition": cond, "seed": args.seed, "gpu": gpu,
                   "started": time.strftime("%H:%M:%S"),
                   "done": [r["retention"] for r in results]}, status_path)

        # --- load + attach a fresh adapter (replace-only: never stacked)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"], quantization_config=bnb, device_map={"": gpu},
            torch_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        torch.manual_seed(args.seed + gen)
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=cfg["rank"], lora_alpha=cfg["rank"] * 2, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))

        # --- train
        def tok_fn(ex):
            return tok(ex["text"], truncation=True, max_length=MAX_LENGTH,
                       padding="max_length")

        train_ds = Dataset.from_dict({"text": prev}).map(
            tok_fn, batched=True, remove_columns=["text"],
            desc="    tokenize", load_from_cache_file=False)
        train_ds.set_format("torch")

        bs, accum = get_batch_size(cfg, gpu)
        targs = TrainingArguments(
            output_dir=str(out_dir / "tmp"), num_train_epochs=EPOCHS,
            per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
            learning_rate=_lr, bf16=True, logging_steps=9999,
            save_strategy="no", report_to="none", seed=args.seed + gen,
            # No group_by_length: removed in transformers 5.x, and it would
            # have been a no-op anyway — padding="max_length" makes every
            # example exactly MAX_LENGTH tokens.
            dataloader_num_workers=2, disable_tqdm=False)

        # Trust the object, not the kwargs: train_batch_size is what the Trainer
        # actually uses, and DataParallel inflates it (n_gpu > 1). This assertion
        # is the last line of defence for the fixed geometry of §5.6.
        eff = targs.train_batch_size * targs.gradient_accumulation_steps
        print(f"    batch {bs} x accum {accum} (effective {eff})  "
              f"n_gpu={targs.n_gpu} dev={targs.device}")
        if targs.n_gpu != 1 or eff != bs * accum:
            raise SystemExit(
                f"Trainer geometry mismatch: n_gpu={targs.n_gpu}, "
                f"train_batch_size={targs.train_batch_size}, effective={eff} "
                f"(wanted {bs*accum}).\n"
                f"  DataParallel is on. Re-run with --gpu N so only one device "
                f"is visible.")
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
        # Peak allocation is the number `min_free_gb` should be derived from. The
        # current values are estimates, and the OOM that led to §5.7 happened
        # under DataParallel (double the batch), so they may be far too
        # conservative — a too-high threshold refuses a card that would have
        # worked fine. Reset here and read AFTER generation (below), since
        # generate() at batch 32 may well peak higher than training does.
        torch.cuda.reset_peak_memory_stats()
        trainer.train()
        train_peak_gb = torch.cuda.max_memory_reserved() / 1024 ** 3
        print(f"    pico de VRAM no treino: {train_peak_gb:.2f} GB "
              f"(min_free_gb={cfg['min_free_gb']:.0f})")
        model.eval()
        del trainer, train_ds
        gc.collect()
        torch.cuda.empty_cache()

        # --- measure
        spectrum = compute_lora_spectrum(model)
        save_json({"gen": gen, "of": args.generations, "phase": "eval_k0",
                   "condition": cond, "seed": args.seed, "gpu": gpu,
                   "done": [r["retention"] for r in results]}, status_path)
        k0_res = evaluate_k0(model, tok, k0_q, k0_a, cfg)
        ret = sum(k0_res)
        print(f"    K0: {ret}/{K0} ({ret/K0:.1%})"
              + (f"   eff_rank={spectrum['eff_rank']:.1f}" if spectrum else ""))

        if adapter_wanted(gen):
            model.save_pretrained(str(out_dir / f"adapter_gen{gen}"))

        # --- generate the next corpus, then apply the regime
        save_json({"gen": gen, "of": args.generations, "phase": "generate",
                   "condition": cond, "seed": args.seed, "gpu": gpu,
                   "done": [r["retention"] for r in results]}, status_path)
        raw = generate_synthetic(model, tok, train_q, cfg,
                                 seed_offset=args.seed + gen + 100)
        div = compute_diversity(raw)
        if active:
            nxt, dose_info = downsample_relative(raw, args.dose,
                                                 args.seed * 1000 + gen)
            print(f"    C5: {dose_info['examples_before']} -> "
                  f"{dose_info['examples_after']} ex, "
                  f"removed {dose_info['removed_frac']:.1%} of tokens")
        else:
            n_tok = sum(len(t.split()) for t in raw)
            nxt = raw
            dose_info = {"tokens_before": n_tok, "tokens_after": n_tok,
                         "removed_frac": 0.0, "examples_before": len(raw),
                         "examples_after": len(raw)}
        print(f"    diversity: len={div.get('mean_length')} "
              f"d1={div.get('distinct_1')} mtld={div.get('mtld')}")
        print(f"    pico de VRAM na geração inteira: "
              f"{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")

        # Atomic: the corpus lands complete or not at all. Written BEFORE the
        # results entry, so a crash between the two re-does one generation
        # rather than orphaning the trajectory.
        syn_path = save_json(nxt, out_dir / f"syn_gen{gen}.json")

        # Whole-generation peak: training, K0 eval, and generation all included.
        # Read before the model is freed, since defrag() resets the allocator.
        peak_gb = torch.cuda.max_memory_reserved() / 1024 ** 3

        del model, raw, nxt
        defrag()

        elapsed = time.time() - t0
        results.append({
            "gen": gen,
            "retention": ret,
            "k0_size": K0,
            "retention_pct": round(100 * ret / K0, 2),
            "all_results": k0_res,
            "intervention_active": active,
            "switch_gen": None if switch == float("inf") else switch,
            "dose": args.dose if active else 0.0,
            "n_examples_in": len(prev),
            "n_tokens_in": sum(len(t.split()) for t in prev),
            "dose_applied": dose_info,
            "diversity": div,
            "adapter": spectrum,
            "time": round(elapsed, 1),
            "train_peak_gb": round(train_peak_gb, 2),
            "peak_gb": round(peak_gb, 2),
        })
        # results.json stays a plain list, matching the ETP output shape;
        # the run configuration lives in a sibling meta.json.
        save_json(results, result_path)
        save_json({
            "condition": cond, "backbone": args.backbone, "model": cfg["model"],
            "rank": cfg["rank"], "seed": args.seed, "lr": _lr, "epochs": EPOCHS,
            "switch_gen": None if switch == float("inf") else switch,
            "dose": args.dose, "dose_mode": "relative",
            "k0_size": K0, "k0_indices_sha": k0_meta["indices_sha"],
            "k0_eval_size": k0_meta["eval_size"], "generations": args.generations,
            "save_adapters": args.save_adapters,
            "gen_batch": cfg["gen_batch"], "eval_batch": cfg.get("eval_batch"),
            # Recorded so a reviewer can verify every arm shared one geometry;
            # these must be identical across conditions (see get_batch_size).
            # effective_batch is read off the TrainingArguments object, so it
            # would expose a DataParallel inflation rather than hide it.
            "train_batch": bs, "train_accum": accum,
            "effective_batch": bs * accum, "n_gpu": 1,
            "gpu_physical": PHYSICAL_GPU,
        }, out_dir / "meta.json")

        # Disk hygiene: 15 corpora x ~50 MB x 26 trajectories would be ~20 GB.
        # Keep the last two — one is the resume point, one is a safety margin.
        if args.keep_corpora == "recent":
            old = out_dir / f"syn_gen{gen-2}.json"
            if old.exists() and gen - 2 >= 1:
                old.unlink()

        eta = (time.time() - t_run) / (gen - start_gen + 1) * (args.generations - gen)
        print(f"    time: {elapsed:.0f}s   eta {eta/3600:.1f}h")

    save_json({"phase": "done", "condition": cond, "seed": args.seed,
               "trajectory": [r["retention"] for r in results]}, status_path)
    traj = ' '.join(str(r['retention']) for r in results)
    print(f"\n  trajectory: {traj}")
    print(f"  total: {(time.time()-t_run)/3600:.1f}h  ->  {result_path}")


if __name__ == "__main__":
    main()
