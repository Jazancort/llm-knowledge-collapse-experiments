#!/usr/bin/env bash
# ============================================================
# Bloco 5 — Opção A: pipeline original g1_rank_ablation.py
# K0~79, eval_size=200, batch=2x8 (mesmos do artigo)
# GPU: 1 (RTX 4000 Ada 20 GB)
#
# A1. C1 multi-seed: r=256, seeds 137+256, 10 gens   (~1.5h)
# A2. Π prospectivo (pipeline original): 3 células    (~1.5h)
#
# Lançar:
#   tmux new -s bloco5
#   bash scripts/revision/run_bloco5_sequential.sh 2>&1 | tee outputs/bloco5_logs/master.log
# ============================================================
set -euo pipefail
cd ~/scratch/llm-knowledge-collapse
GPU=1
export CUDA_VISIBLE_DEVICES=$GPU
LOG_DIR="outputs/bloco5_logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/master.log"; }
run_exp() {
    local label="$1"; shift
    log "START  $label"
    uv run python "$@" 2>&1 | tee "$LOG_DIR/${label}.log"
    local rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ]; then
        log "DONE   $label (exit 0)"
    else
        log "FAIL   $label (exit $rc) — abortando"
        exit $rc
    fi
}

log "=== BLOCO 5 INICIO (Opção A — pipeline original K0~79) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader

# A1. C1 multi-seed (r=256, lr=1e-5 default)
run_exp "a1_C1_r256_seed137" \
    scripts/g1_rank_ablation.py --rank 256 --seed 137 --generations 10

run_exp "a1_C1_r256_seed256" \
    scripts/g1_rank_ablation.py --rank 256 --seed 256 --generations 10

# A2. Π prospectivo — pré-registrado em ee0aecc9
run_exp "a2_pi_r128_lr5e6_seed15" \
    scripts/g1_rank_ablation.py --rank 128 --lr 5e-6 --seed 15 \
        --tag "_lr5e6" --generations 10

run_exp "a2_pi_r128_lr2e5_seed15" \
    scripts/g1_rank_ablation.py --rank 128 --lr 2e-5 --seed 15 \
        --tag "_lr2e5" --generations 10

run_exp "a2_pi_r32_lr2e5_seed15" \
    scripts/g1_rank_ablation.py --rank 32 --lr 2e-5 --seed 15 \
        --tag "_lr2e5" --generations 10

log "=== BLOCO 5 CONCLUIDO ==="
log "Outputs: outputs/g1_rank*"
