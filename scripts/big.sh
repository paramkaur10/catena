#!/usr/bin/env bash
# The big run: Clarity on Sentinel-2, Landsat 8, HEO and PeruSat-1, the versions real and sim
# side by side (real on cuda:0, sim on cuda:1; catena.train_clarity), seeds 0, 1 and 2, one
# after the other.
#
#     bash scripts/big.sh                         (inside tmux, from ~/projects/catena)
#     STEPS=30000 bash scripts/big.sh             (fewer steps per seed)
#     SEEDS="1" bash scripts/big.sh               (only some seeds)
#     OUT=/root/catena_data/runs/big2 bash scripts/big.sh --shares 0.25 0.25 0.25 0.25
#                                                 (another folder; options for the training)
#     OUT=/root/catena_data/runs/big2 SEEDS="3 4 5" bash scripts/big.sh \
#         --q-weights 0.5 0.3 0.2 --faults 0.5    (the second run: sim with more 10 m and
#                                                 with calibration faults; real as before)
#
# Each seed writes OUT/seed<N>/ (train.log, config.json, metrics.jsonl, last.pt, final.pt);
# OUT/big.log says when each seed started, with which options, and how it ended. Start it
# again after a stop: a seed with final.pt is skipped, one with last.pt goes on from it.
set -u
DATA=${DATA:-/root/catena_data}
OUT=${OUT:-$DATA/runs/big}
STEPS=${STEPS:-60000}
mkdir -p "$OUT"
for SEED in ${SEEDS:-0 1 2}; do
    RUN="$OUT/seed$SEED"
    if [ -f "$RUN/final.pt" ]; then
        echo "$(date -u +%FT%TZ) seed $SEED: already done" | tee -a "$OUT/big.log"
        continue
    fi
    mkdir -p "$RUN"
    echo "$(date -u +%FT%TZ) seed $SEED: start, $STEPS steps $*" | tee -a "$OUT/big.log"
    python -m catena.train_clarity "$RUN" --data "$DATA" --seed "$SEED" --steps "$STEPS" "$@" \
        >> "$RUN/train.log" 2>&1
    echo "$(date -u +%FT%TZ) seed $SEED: exit $?" | tee -a "$OUT/big.log"
done
