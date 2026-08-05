#!/bin/bash
# Train with automatic resume after an OOM kill.
#
# A PyTorch MPS process needs ~1.1 GB here regardless of grid/batch/pool size --
# that floor is runtime overhead, not our tensors, so it cannot be configured away.
# When free system RAM drops below it, macOS kills the training. Two comm runs died
# that way, one at step 1050 and one at step 1.
#
# Rather than fight for memory, this leans on the bit-exact resume from C4: frequent
# checkpoints mean a kill costs at most `ckpt_interval` steps, and the run picks up
# exactly where it left off (same weights, optimiser moments, LR schedule, and all
# four RNG streams).
#
#   ./scripts/train_resilient.sh configs/comm.yaml runs/comm 12

set -u
CONFIG="${1:?usage: train_resilient.sh <config> <run_dir> [max_attempts]}"
RUN_DIR="${2:?usage: train_resilient.sh <config> <run_dir> [max_attempts]}"
MAX_ATTEMPTS="${3:-12}"
PY=.venv/bin/python

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    CKPT="$RUN_DIR/ckpt/last.pt"
    if [ -f "$CKPT" ]; then
        STEP=$($PY -c "import torch;print(torch.load('$CKPT',map_location='cpu',weights_only=False)['step'])" 2>/dev/null || echo "?")
        echo "[attempt $attempt] resuming from step $STEP"
        $PY scripts/train.py --config "$CONFIG" --run-dir "$RUN_DIR" --resume "$CKPT"
    else
        echo "[attempt $attempt] starting fresh"
        $PY scripts/train.py --config "$CONFIG" --run-dir "$RUN_DIR"
    fi

    code=$?
    if [ $code -eq 0 ]; then
        echo "[done] training completed"
        exit 0
    fi
    # 137/143/144 = SIGKILL/SIGTERM, i.e. the OOM killer. Anything else is a real
    # error and retrying would just loop on the same bug.
    if [ $code -ne 137 ] && [ $code -ne 143 ] && [ $code -ne 144 ]; then
        echo "[abort] exit $code is not an OOM kill -- not retrying"
        exit $code
    fi
    echo "[killed] exit $code, likely OOM; retrying"
    sleep 5
done

echo "[abort] exhausted $MAX_ATTEMPTS attempts"
exit 1
