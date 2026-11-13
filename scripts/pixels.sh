#!/usr/bin/env bash
# Coarser pixels: the big run again (arms real and sim, seeds 0-2, Sentinel-2, Landsat 8, HEO
# and PeruSat-1 as in runs/big), with half of the batches from Landsat 8, HEO and PeruSat-1
# shown at a pixel 2 or 3 times larger and the card told so (--coarser 0.5). Beside runs/big,
# which drew the same samples otherwise: what one pixel size per satellite cost. It waits for
# any training or evaluation still running, then:
#   train   OUT/seed0-2 (scripts/big.sh)
#   then    GPU 0: the two on Tests A-F: tests.csv
#           GPU 1: on Tests D-F with the scenes predicted on a 10 m grid (g10.csv) and on a
#           5 m grid (g5.csv: PeruSat-1 at 5 m, Phi-sat-2 nearly as delivered)
#   last    results4.tgz in the project folder: every CSV, log, config and metrics of OUT
#
#     bash scripts/pixels.sh        (inside tmux, from ~/projects/catena)
#
# OUT (DATA/runs/pixels) holds pixels.log, which says when each step started and how it
# ended. Started again, it goes on: a seed from its last.pt, and a CSV already written is not
# made again (delete one that a failed step left).
set -u
DATA=${DATA:-/root/catena_data}
RUNS=$DATA/runs
OUT=${OUT:-$RUNS/pixels}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/pixels.log"; }

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

say "pixels: start (after any job still running)"
quiet "catena[.]train_clarity" "catena[.]nas" "catena[.]evaluate_clarity" "catena[.]onboard"
say "training: start (scripts/big.sh: real and sim, seeds 0-2, coarser pixels for half)"
OUT="$OUT" DATA="$DATA" bash scripts/big.sh --arms real sim --coarser 0.5
say "training: done (see $OUT/big.log)"

if finished "$OUT"; then
    evaluate "$OUT/tests.csv" "$OUT/tests.log" "$OUT" --device cuda:0 &
    {
        evaluate "$OUT/g10.csv" "$OUT/g10.log" "$OUT" --tests D E F --grid 10 --device cuda:1
        evaluate "$OUT/g5.csv" "$OUT/g5.log" "$OUT" --tests D E F --grid 5 --device cuda:1
    } &
    wait
else
    say "not every seed finished: nothing scored (see $OUT/big.log and the seeds' train.log)"
fi

say "packing: $HERE/results4.tgz"
(
    cd "$(dirname "$OUT")" && find "$(basename "$OUT")" -maxdepth 2 \
        \( -name '*.csv' -o -name '*.log' -o -name config.json -o -name metrics.jsonl \) \
        | tar -czf "$HERE/results4.tgz" -T -
)
say "pixels: done"
