#!/usr/bin/env bash
# The told-versus-blind pilot: seeds 0, 1 and 2, one after the other. Each run trains both
# arms on the same batches, told on cuda:0 and blind on cuda:1 (catena.train).
#
#     bash scripts/pilot.sh                       (inside tmux, from ~/projects/catena)
#     STEPS=20000 bash scripts/pilot.sh           (fewer steps per seed)
#
# Each seed writes OUT/seed<N>/ (train.log, config.json, metrics.jsonl, last.pt, final.pt);
# OUT/pilot.log says when each seed started and how it ended. A seed that already has
# final.pt is skipped, so the script can be started again after a stop.
set -u
CHIPS=${CHIPS:-/root/catena_data/chips/cloudsen12plus}
OUT=${OUT:-/root/catena_data/runs/pilot}
STEPS=${STEPS:-30000}
mkdir -p "$OUT"
for SEED in 0 1 2; do
    RUN="$OUT/seed$SEED"
    if [ -f "$RUN/final.pt" ]; then
        echo "$(date -u +%FT%TZ) seed $SEED: already done" | tee -a "$OUT/pilot.log"
        continue
    fi
    mkdir -p "$RUN"
    echo "$(date -u +%FT%TZ) seed $SEED: start, $STEPS steps" | tee -a "$OUT/pilot.log"
    python -m catena.train "$CHIPS" "$RUN" --seed "$SEED" --steps "$STEPS" >> "$RUN/train.log" 2>&1
    echo "$(date -u +%FT%TZ) seed $SEED: exit $?" | tee -a "$OUT/pilot.log"
done
