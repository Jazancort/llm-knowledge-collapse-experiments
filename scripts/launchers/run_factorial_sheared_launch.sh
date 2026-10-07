#!/usr/bin/env bash
# run_factorial_sheared_launch.sh
# Factorial 2x2 Sheared-LLaMA-1.3B, 8 seeds, retry em OOM.
#
# GPU 1 (RTX 4000 Ada, 20 GB) — seeds 42 137 256 15
# GPU 0 (RTX 5000 Ada, 32 GB) — seeds 77  99 101 202
#
# Uso:  bash scripts/launchers/run_factorial_sheared_launch.sh [--dry-run]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/../.."

SCRIPT="scripts/revision/run_factorial_n8.py"
OUTPUT="outputs/factorial_n8_sheared"
MIN_FREE_GB=11
POLL_INTERVAL=60
MAX_RETRIES=3
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID

free_gb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
               --id="$1" | awk '{print int($1/1024)}'
}

wait_for_vram() {
    local gpu=$1 min=$2 label=$3
    while true; do
        local f; f=$(free_gb "$gpu")
        if [ "$f" -ge "$min" ]; then
            echo "[launcher] GPU $gpu: ${f} GB livres — lançando $label"; return 0
        fi
        echo "[launcher] GPU $gpu: ${f} GB (preciso $min) — aguardando $label..."
        sleep "$POLL_INTERVAL"
    done
}

seed_done() { [ -f "${OUTPUT}/seed${1}/results_all.json" ]; }

run_seed() {
    local seed=$1 gpu=$2
    local log="${OUTPUT}/seed${seed}/run.log"
    mkdir -p "${OUTPUT}/seed${seed}"

    for attempt in 1 2 3; do
        echo "[s${seed} g${gpu}] Tentativa ${attempt}/${MAX_RETRIES}  $(date)" | tee -a "$log"
        uv run python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

        set +e
        uv run python "$SCRIPT" \
            --backbone sheared --seed "$seed" --gpu "$gpu" --output "$OUTPUT" \
            2>&1 | tee -a "$log"
        local rc=${PIPESTATUS[0]}
        set -e

        if [ $rc -eq 0 ]; then
            echo "[s${seed} g${gpu}] CONCLUÍDO  $(date)" | tee -a "$log"; return 0
        fi

        if grep -qE "CUDA out of memory|OutOfMemoryError|out of memory" "$log" 2>/dev/null; then
            echo "[s${seed} g${gpu}] OOM — aguardando 60 s..." | tee -a "$log"
            sleep 60
            uv run python3 -c "import gc,torch; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true
        else
            echo "[s${seed} g${gpu}] ERRO (rc=$rc, não-OOM) — abortando" | tee -a "$log"
            return $rc
        fi
    done

    echo "[s${seed} g${gpu}] Falhou após ${MAX_RETRIES} tentativas." | tee -a "$log"
    return 1
}

worker() {
    local gpu=$1; shift
    for seed in "$@"; do
        if seed_done "$seed"; then
            echo "[launcher] seed $seed: já concluído — a saltar"; continue
        fi
        wait_for_vram "$gpu" "$MIN_FREE_GB" "seed $seed"
        [ $DRY_RUN -eq 1 ] && { echo "[dry-run] run_seed $seed $gpu"; continue; }
        run_seed "$seed" "$gpu"
        echo "[launcher] seed $seed GPU $gpu: concluído."
    done
    echo "[launcher] Worker GPU $gpu: todos os seeds tratados."
}

mkdir -p outputs/logs "$OUTPUT"
echo "=== Sheared factorial launch  $(date) ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

if [ $DRY_RUN -eq 1 ]; then
    worker 1 42 137 256 15
    worker 0 77 99 101 202
    exit 0
fi

# Exportar funções para as sub-shells tmux
export -f free_gb wait_for_vram seed_done run_seed worker
export OUTPUT SCRIPT MIN_FREE_GB POLL_INTERVAL MAX_RETRIES DRY_RUN

tmux new-session -d -s sheared_g1 \
    "export CUDA_DEVICE_ORDER=PCI_BUS_ID; cd $(pwd); \
     worker 1 42 137 256 15 \
     2>&1 | tee outputs/logs/sheared_worker_gpu1.log; \
     echo '=== GPU1 WORKER DONE ==='; sleep 60"

tmux new-session -d -s sheared_g0 \
    "export CUDA_DEVICE_ORDER=PCI_BUS_ID; cd $(pwd); \
     worker 0 77 99 101 202 \
     2>&1 | tee outputs/logs/sheared_worker_gpu0.log; \
     echo '=== GPU0 WORKER DONE ==='; sleep 60"

echo ""
echo "Workers lançados:"
tmux ls | grep sheared
echo ""
echo "Monitorizar:"
echo "  tmux attach -t sheared_g1        # GPU 1 (seeds 42 137 256 15)"
echo "  tmux attach -t sheared_g0        # GPU 0 (seeds 77 99 101 202)"
echo "  watch -n 60 'cat outputs/factorial_n8_sheared/seed*/run.log | grep -E \"Tentativa|CONCLUÍDO|ERRO|OOM|Gen 12\"'"
