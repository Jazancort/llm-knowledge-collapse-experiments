#!/bin/bash
# Auto-restart wrapper: keeps running fft_drift_gen10.py until it completes all seeds.
# The script has incremental save, so each restart resumes where it left off.

cd ~/scratch/llm-knowledge-collapse

while true; do
    echo "$(date): Starting fft_drift_gen10.py..."
    uv run python scripts/fft_drift_gen10.py
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date): Script completed successfully!"
        break
    fi
    
    echo "$(date): Script died (exit $EXIT_CODE). Waiting 10s then restarting..."
    sleep 10
done
