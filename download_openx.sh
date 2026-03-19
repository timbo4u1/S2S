#!/bin/bash
# Infinite retry download for Open-X shard
# Resumes automatically on every failure

URL="https://huggingface.co/datasets/jxu124/OpenX-Embodiment/resolve/main/roboturk/roboturk_00000.tar"
OUT="$HOME/S2S/openx_data/roboturk_00000.tar"
mkdir -p ~/S2S/openx_data

echo "Downloading roboturk_00000.tar (~717MB)"
echo "Will retry forever. Ctrl+C to stop."
echo ""

ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[attempt $ATTEMPT] $(date '+%H:%M:%S')"
    
    curl \
        --location \
        --continue-at - \
        --retry 999 \
        --retry-delay 5 \
        --retry-max-time 0 \
        --connect-timeout 30 \
        --max-time 3600 \
        --progress-bar \
        --output "$OUT" \
        "$URL"
    
    EXIT=$?
    if [ $EXIT -eq 0 ]; then
        SIZE=$(du -h "$OUT" | cut -f1)
        echo ""
        echo "✅ Done — $OUT ($SIZE)"
        break
    fi
    
    echo "⚠ Interrupted (exit=$EXIT) — waiting 10s then resuming..."
    sleep 10
done
