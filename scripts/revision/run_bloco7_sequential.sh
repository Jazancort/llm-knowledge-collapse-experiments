#!/usr/bin/env bash
# ============================================================
# Bloco 7 — Dose-response de exposição + diagnóstico de pipeline
# Pre-registered: a916ebf (2026-09-30T20:40 UTC-3)
# Pipeline: g1_rank_ablation.py, r=256, lr=1e-5, K0~79
# GPU: 1 | Ordem: 7a → 7c → 7b
#
# 7a: Dose-response (0/5/10/25/50%) × 5 seeds (15/137/256/42/77)
#     doses 0% e 5% seeds 15/137/256 já existem → resume pula
# 7c: LR equivalência — dose=50% vs dose=0/lr=5e-6 (previsão 89-92%)
# 7b: Steps-matched — dose=25% com max_steps=250 (igual ao dose=0)
#
# Lançar:
#   tmux new -s bloco7
#   bash scripts/revision/run_bloco7_sequential.sh 2>&1 | tee outputs/bloco7_logs/master.log
# ============================================================
set -euo pipefail
cd ~/scratch/llm-knowledge-collapse
GPU=1
export CUDA_VISIBLE_DEVICES=$GPU
LOG_DIR="outputs/bloco7_logs"
mkdir -p "$LOG_DIR"
R="scripts/g1_rank_ablation.py"
GENS=10

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/master.log"; }
run_exp() {
    local label="$1"; shift
    log "START  $label"
    uv run python "$@" 2>&1 | tee "$LOG_DIR/${label}.log"
    local rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ]; then log "DONE   $label (exit 0)"
    else log "FAIL   $label (exit $rc) — abortando"; exit $rc; fi
}

log "=== BLOCO 7 INICIO — prereg a916ebf ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader

# ── 7a: Dose-response ─────────────────────────────────────────────────────────
# Doses: 0(0%), 05(5%), 10(10%), 25(25%), 50(50%)
# Seeds: 15, 137, 256, 42, 77
# dose=0%  seeds 15/137/256 → dirs g1_rank256_seed{s}          (existem: resume pula)
# dose=5%  seeds 15/137/256 → dirs g1_rank256_seed{s}_c5d05    (existem: resume pula)
# Novos: dose=10/25/50 × todos seeds + dose=0/5 × seeds 42/77

log "--- 7a: Dose-response ---"
for SEED in 15 137 256 42 77; do
  run_exp "7a_dose00_s${SEED}" $R --rank 256 --seed $SEED --dose 0.00 \
    --tag ""         --generations $GENS
  run_exp "7a_dose05_s${SEED}" $R --rank 256 --seed $SEED --dose 0.05 \
    --tag "_c5d05"   --generations $GENS
  run_exp "7a_dose10_s${SEED}" $R --rank 256 --seed $SEED --dose 0.10 \
    --tag "_dose10"  --generations $GENS
  run_exp "7a_dose25_s${SEED}" $R --rank 256 --seed $SEED --dose 0.25 \
    --tag "_dose25"  --generations $GENS
  run_exp "7a_dose50_s${SEED}" $R --rank 256 --seed $SEED --dose 0.50 \
    --tag "_dose50"  --generations $GENS
done

# ── 7c: LR equivalência (previsão pré-registrada) ────────────────────────────
log "--- 7c: LR equivalência dose=50% vs lr=5e-6 ---"
for SEED in 15 137 256; do
  # dose=50% já rodou no 7a — resume pula
  run_exp "7c_dose50_s${SEED}" $R --rank 256 --seed $SEED --dose 0.50 \
    --tag "_dose50"  --generations $GENS
  # dose=0%, lr=5e-6 (reference; seed15 existe no bloco5)
  run_exp "7c_lr5e6_s${SEED}"  $R --rank 256 --seed $SEED --dose 0.00 \
    --lr 5e-6 --tag "_lr5e6"   --generations $GENS
done

# ── 7b: Steps-matched control ─────────────────────────────────────────────────
# dose=0% → 250 steps. dose=25% → ~187 steps. Forçar dose=25% a 250 steps.
log "--- 7b: Steps-matched (dose=25%, max_steps=250) ---"
for SEED in 15 137 256; do
  run_exp "7b_dose25_matched_s${SEED}" $R --rank 256 --seed $SEED --dose 0.25 \
    --steps 250 --tag "_dose25_matched" --generations $GENS
done

log "=== BLOCO 7 CONCLUIDO ==="
