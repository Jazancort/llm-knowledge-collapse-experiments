#!/usr/bin/env bash
# ============================================================
# Revisão Final — G1 → G2 → G3 → G4 → G5
# Pre-registered: 3b1712b (2026-10-02T09:35 UTC-3)
# Pipeline: g1_rank_ablation.py, commit c3ba205
# GPU: 1
#
# G1: Qwen r=256 dose=0% seeds 15/137/256  (~3h)  R: pipeline consistency
# G2: Gemma3 r=10  dose 0/10/25/50% ×3s    (~5h)  R1.Q4
# G3: Qwen bf16 r=16+r=256 ×3 seeds       (~7h)  R1.Q2 quant
# G4: Qwen r=256 order ×3 shuffle seeds   (~2h)  R1.Q2 order
# G5: Pi prospective 3 cells               (~1.5h) R1.Q1
# ============================================================
set -euo pipefail
cd ~/scratch/llm-knowledge-collapse
GPU=1
export CUDA_VISIBLE_DEVICES=$GPU
LOG_DIR="outputs/revisao_final_logs"
INDEX="outputs/runs_index.csv"
mkdir -p "$LOG_DIR"

# Criar cabeçalho do runs_index.csv se não existir
if [ ! -f "$INDEX" ]; then
    echo "label,exp,backbone,rank,seed,dose,lr,no_quant,tag,timestamp_start,exit_code" > "$INDEX"
fi

R="scripts/g1_rank_ablation.py"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/master.log"; }

run_exp() {
    local label="$1"; shift
    local exp="$1"; shift
    log "START  $label"
    local ts=$(date '+%Y-%m-%dT%H:%M:%S')
    uv run python "$@" 2>&1 | tee "$LOG_DIR/${label}.log"
    local rc=${PIPESTATUS[0]}
    echo "${label},${exp},,,,,,,,$ts,$rc" >> "$INDEX"
    if [ $rc -eq 0 ]; then log "DONE   $label (exit 0)"
    else log "FAIL   $label (exit $rc)"; exit $rc; fi
}

log "=== REVISAO FINAL INICIO — prereg 3b1712b — commit c3ba205 ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# ── G1: Qwen r=256 dose=0% no commit c3ba205 ────────────────────────
log "--- G1: Qwen r=256 dose=0%, seeds 15/137/256 (pipeline consistency) ---"
for SEED in 15 137 256; do
    run_exp "G1_qwen_r256_s${SEED}_dose00" "G1" \
        $R --rank 256 --seed $SEED --dose 0.00 \
        --tag "_G1_dose00" --generations 10
done

# ── G2: Gemma3 r=10 dose-response ───────────────────────────────────
log "--- G2: Gemma3 r=10, doses 0/10/25/50%, seeds 15/137/256 ---"
GEMMA="google/gemma-3-1b-it"
for SEED in 15 137 256; do
    for DOSE in 0.00 0.10 0.25 0.50; do
        DTAG=$(printf "%.0f" $(echo "$DOSE * 100" | bc))
        run_exp "G2_gemma_r10_s${SEED}_dose${DTAG}" "G2" \
            $R --model $GEMMA --rank 10 --seed $SEED --dose $DOSE \
            --tag "_G2_dose${DTAG}" --generations 10
    done
done

# ── G3: LoRA bf16 sem NF4 ───────────────────────────────────────────
log "--- G3: Qwen bf16 no-NF4, r=16 e r=256, seeds 15/137/256 ---"
for SEED in 15 137 256; do
    run_exp "G3_qwen_r16_s${SEED}_bf16"  "G3" \
        $R --rank 16  --seed $SEED --no-quant \
        --tag "_G3_bf16" --generations 10
    run_exp "G3_qwen_r256_s${SEED}_bf16" "G3" \
        $R --rank 256 --seed $SEED --no-quant \
        --tag "_G3_bf16" --generations 10
done

# ── G4: Ordem dos dados ──────────────────────────────────────────────
# Usa --seed para controlar o shuffle do dataset (seed do modelo fixo em 15)
log "--- G4: Qwen r=256, init_seed=15, shuffle seeds 42/77/256 ---"
for SHUF in 42 77 256; do
    run_exp "G4_qwen_r256_s15_shuf${SHUF}" "G4" \
        $R --rank 256 --seed $SHUF \
        --tag "_G4_shuf${SHUF}" --generations 5
done

# ── G5: Pi prospectivo (células pré-registradas ee0aecc9) ───────────
log "--- G5: Pi prospectivo, pipeline g1 ---"
run_exp "G5_qwen_r128_s15_lr5e6"  "G5" \
    $R --rank 128 --seed 15 --lr 5e-6  --tag "_G5_lr5e6"  --generations 5
run_exp "G5_qwen_r128_s15_lr2e5"  "G5" \
    $R --rank 128 --seed 15 --lr 2e-5  --tag "_G5_lr2e5"  --generations 5
run_exp "G5_qwen_r32_s15_lr2e5"   "G5" \
    $R --rank 32  --seed 15 --lr 2e-5  --tag "_G5_lr2e5"  --generations 5

log "=== REVISAO FINAL CONCLUIDA ==="
