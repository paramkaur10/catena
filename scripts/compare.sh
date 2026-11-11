#!/usr/bin/env bash
# More than thresholds? Every network beside the other models and a floor of thresholds, on the
# same pixels, and the check that tells a network that reads around a pixel from a rule of each
# pixel alone:
#   first  the thresholds (catena.threshold), fitted on the validation cameras, unless
#          runs/threshold/threshold.json is there
#   GPU 0  every arm of every seed (runs/big, runs/arms, runs/nas_final) on Tests A-F beside
#          OmniCloudMask (run on B and C too), SEnSeIv2 (B-F) and the thresholds: all.csv
#   GPU 1  seed 0 of Clarity sim and real, of the two found trunks and of the fixed and blind
#          arms, as they are and with their pixels scrambled, the thresholds beside:
#          scrambled.csv
#
#     bash scripts/compare.sh          (inside tmux, from ~/projects/catena)
#
# OUT (DATA/runs/compare) holds compare.log, which says when each step started and how it
# ended, the two CSVs and each step's log. Started again, a CSV already written is not made
# again (delete one that a failed step left).
set -u
DATA=${DATA:-/root/catena_data}
RUNS=$DATA/runs
OUT=${OUT:-$RUNS/compare}
FLOOR=$RUNS/threshold/threshold.json
mkdir -p "$OUT"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/compare.log"; }

evaluate() {  # evaluate CSV LOG [catena.evaluate_clarity arguments]
    local csv=$1 log=$2
    shift 2
    if [ -f "$csv" ]; then
        say "$(basename "$log" .log): $csv is there already"
        return
    fi
    say "$(basename "$log" .log): start"
    python -u -m catena.evaluate_clarity "$@" --out "$csv" >> "$log" 2>&1
    say "$(basename "$log" .log): exit $?"
}

if [ -f "$FLOOR" ]; then
    say "thresholds: $FLOOR is there already"
else
    say "thresholds: start"
    python -u -m catena.threshold "$(dirname "$FLOOR")" --data "$DATA" >> "$OUT/threshold.log" 2>&1
    say "thresholds: exit $?"
fi
BESIDE=()
if [ -f "$FLOOR" ]; then
    BESIDE=(--threshold "$FLOOR")
fi
evaluate "$OUT/all.csv" "$OUT/all.log" "$RUNS/big" "$RUNS/arms" "$RUNS/nas_final" \
    --data "$DATA" --ocm --sensei "${BESIDE[@]}" --device cuda:0 &
evaluate "$OUT/scrambled.csv" "$OUT/scrambled.log" "$RUNS/big/seed0" "$RUNS/nas_final/seed0" \
    "$RUNS/arms/fixed/seed0" --data "$DATA" --cards honest "pixels scrambled" "${BESIDE[@]}" \
    --device cuda:1 &
wait
say "compare: done"
