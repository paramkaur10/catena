#!/usr/bin/env bash
# The flight model: the descriptor head with the two searched trunks (arms nas and nas2, the
# codes runs/nas_final trained), made for the Hailo-8 (--hailo: BatchNorm, its statistics
# frozen before training ends so they fold into the convolutions; ReLU; nearest-neighbour
# upsampling), seeds 0-2, with the coarser pixels of runs/nas_coarser (--coarser 0.5): the
# same run as runs/nas_coarser but for the layers, so the two say what the layers the chip
# takes cost. It claims the GPUs from scripts/final.sh after its stage 1 (it creates
# RUNS/final_stage2.skip), waits for any training or evaluation still running, then:
#   train   OUT/seed0-2 (scripts/big.sh)
#   then    GPU 0: Tests A-F (tests.csv), then the 8-bit twins on Tests B-F (int8.csv)
#           GPU 1: Tests D-F on a 10 m and on a 5 m grid (g10.csv, g5.csv)
#   then    what flies, with nothing else running (catena.onboard): onboard.csv
#   last    results5.tgz in the project folder, then scripts/light.sh (Sentinel-2's windows
#           by the sun, results6.tgz)
#
#     bash scripts/flight.sh        (inside tmux, from ~/projects/catena)
#
# OUT (DATA/runs/flight) holds flight.log, which says when each step started and how it ended.
# Started again, it goes on: a seed from its last.pt, and a CSV already written is not made
# again (delete one that a failed step left).
set -u
DATA=${DATA:-/root/catena_data}
RUNS=$DATA/runs
OUT=${OUT:-$RUNS/flight}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
touch "$RUNS/final_stage2.skip"  # scripts/final.sh stops after its stage 1

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/flight.log"; }

running() {  # is a python process whose command line matches the pattern running?
    local pid
    for pid in $(pgrep -f "$1"); do
        case "$(cat "/proc/$pid/comm" 2> /dev/null)" in
            python*) return 0 ;;
        esac
    done
    return 1
}

quiet() {  # wait until no python process matches any pattern, twice in a row a minute apart
    local free=0 busy p told=0
    while [ "$free" -lt 2 ]; do
        busy=0
        for p in "$@"; do
            if running "$p"; then
                busy=1
                if [ "$told" -eq 0 ]; then say "waiting for a running $p"; fi
                told=1
            fi
        done
        if [ "$busy" -eq 1 ]; then free=0; else free=$((free + 1)); fi
        if [ "$free" -lt 2 ]; then sleep 60; fi
    done
}

evaluate() {  # evaluate CSV LOG [catena.evaluate_clarity arguments]
    local csv=$1 log=$2
    shift 2
    if [ -f "$csv" ]; then
        say "$(basename "$log" .log): $csv is there already"
        return
    fi
    say "$(basename "$log" .log): start"
    python -u -m catena.evaluate_clarity "$@" --data "$DATA" --out "$csv" >> "$log" 2>&1
    say "$(basename "$log" .log): exit $?"
}

finished() {  # finished FOLDER: has every seed of the run its final.pt?
    local seed
    for seed in ${SEEDS:-0 1 2}; do
        if [ ! -f "$1/seed$seed/final.pt" ]; then return 1; fi
    done
    return 0
}

say "flight: start (after any job still running; final.sh will stop after its stage 1)"
quiet "catena[.]train_clarity" "catena[.]nas" "catena[.]evaluate_clarity" "catena[.]onboard"
TRUNKS="import json, sys; print(' '.join(json.load(open(sys.argv[1]))['args']['trunk']))"
CODES=$(python -c "$TRUNKS" "$RUNS/nas_final/seed0/config.json")
if [ -z "$CODES" ] || [ "$(echo "$CODES" | wc -w)" -ne 2 ]; then
    say "flight: no two trunk codes in $RUNS/nas_final/seed0/config.json, nothing trained"
    exit 1
fi
say "training: start (scripts/big.sh --arms nas nas2 --trunk $CODES --coarser 0.5 --hailo)"
# shellcheck disable=SC2086  # the two codes are two words
OUT="$OUT" DATA="$DATA" bash scripts/big.sh --arms nas nas2 --trunk $CODES --coarser 0.5 --hailo
say "training: done (see $OUT/big.log)"

if finished "$OUT"; then
    {
        evaluate "$OUT/tests.csv" "$OUT/tests.log" "$OUT" --device cuda:0
        evaluate "$OUT/int8.csv" "$OUT/int8.log" "$OUT" --tests B C D E F --int8 --device cuda:0
    } &
    {
        evaluate "$OUT/g10.csv" "$OUT/g10.log" "$OUT" --tests D E F --grid 10 --device cuda:1
        evaluate "$OUT/g5.csv" "$OUT/g5.log" "$OUT" --tests D E F --grid 5 --device cuda:1
    } &
    wait
    if [ -f "$OUT/onboard.csv" ]; then
        say "onboard: $OUT/onboard.csv is there already"
    else
        say "onboard: start (with nothing else running)"
        python -u -m catena.onboard "$OUT" --device cuda:0 --out "$OUT/onboard.csv" \
            >> "$OUT/onboard.log" 2>&1
        say "onboard: exit $?"
    fi
else
    say "not every seed finished: nothing scored (see $OUT/big.log and the seeds' train.log)"
fi

(
    cd "$RUNS" && find "$(basename "$OUT")" -maxdepth 2 \
        \( -name '*.csv' -o -name '*.log' -o -name config.json -o -name metrics.jsonl \) \
        | tar -czf "$HERE/results5.tgz" -T -
)
say "packed: $HERE/results5.tgz"
say "flight: done; Sentinel-2 by the sun next (scripts/light.sh)"
bash scripts/light.sh
