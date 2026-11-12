#!/usr/bin/env bash
# As delivered: a run that never sees a made-up camera or HEO, and what a camera's SNR is worth
# when it is acted on.
#   beside  GPU 0, while the training runs: every arm of every seed so far (runs/big, runs/arms,
#           runs/nas_final) on Test A as it is and with each window averaged 2 x 2 and 3 x 3
#           before the network (binned x2, binned x3): binned.csv, copied to the project
#           folder as soon as it is written (about an hour)
#   both    arms real and chain_real, seeds 0-2 (scripts/big.sh), on Sentinel-2, Landsat 8 and
#           PeruSat-1 as delivered (--shares 0.5 0.25 0 0.25): HEO becomes a satellite never
#           seen, and the MTF and the SNR are told only where a product's documents state them
#   then    GPU 0: the two on Tests A-F: tests.csv
#           GPU 1: their card checks: cards_A.csv (Test A, 200 windows per camera),
#           cards_B.csv (Landsat 8, its noise said 3 times larger) and cards_C.csv (Landsat 7,
#           its pixel said 3 times larger and 3 times smaller)
#   last    results2.tgz in the project folder: every CSV, log, config and metrics of OUT
#
#     bash scripts/delivered.sh        (inside tmux, from ~/projects/catena)
#     STEPS=30000 bash scripts/delivered.sh        (fewer steps per seed)
#
# OUT (DATA/runs/delivered) holds delivered.log, which says when each step started and how it
# ended, each seed's folder and each step's CSV and log. It waits for any training, search or
# evaluation still running. Started again, it goes on: a seed from its last.pt, and a CSV
# already written is not made again (delete one that a failed step left).
set -u
DATA=${DATA:-/root/catena_data}
RUNS=$DATA/runs
OUT=${OUT:-$RUNS/delivered}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/delivered.log"; }

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

binning() {
    evaluate "$OUT/binned.csv" "$OUT/binned.log" "$RUNS/big" "$RUNS/arms" "$RUNS/nas_final" \
        --tests A --cards honest "binned x2" "binned x3" --device cuda:0
    if [ -f "$OUT/binned.csv" ]; then
        cp "$OUT/binned.csv" "$HERE/binned.csv"
        say "binned: copied to $HERE/binned.csv"
    fi
}

say "as delivered: start (after any job still running)"
quiet "catena[.]train_clarity" "catena[.]nas" "catena[.]evaluate_clarity" "catena[.]onboard"
binning &
say "training: start (scripts/big.sh: real and chain_real, seeds 0-2, no HEO, no made-up camera)"
OUT="$OUT" DATA="$DATA" bash scripts/big.sh --arms real chain_real --shares 0.5 0.25 0 0.25
say "training: done (see $OUT/big.log)"
wait

READY=1
for SEED in ${SEEDS:-0 1 2}; do
    if [ ! -f "$OUT/seed$SEED/final.pt" ]; then READY=0; fi
done
if [ "$READY" -eq 1 ]; then
    evaluate "$OUT/tests.csv" "$OUT/tests.log" "$OUT" --device cuda:0 &
    {
        evaluate "$OUT/cards_A.csv" "$OUT/cards_A.log" "$OUT" --tests A --windows 200 \
            --cards honest shuffled "pixel x3" "pixel /3" "wavelengths only" "noise x3" "blur x2" \
            --device cuda:1
        evaluate "$OUT/cards_B.csv" "$OUT/cards_B.log" "$OUT" --tests B --cards "noise x3" \
            --device cuda:1
        evaluate "$OUT/cards_C.csv" "$OUT/cards_C.log" "$OUT" --tests C \
            --cards "pixel x3" "pixel /3" --device cuda:1
    } &
    wait
else
    say "not every seed finished: nothing scored (see $OUT/big.log and the seeds' train.log)"
fi

(
    cd "$(dirname "$OUT")" && find "$(basename "$OUT")" -maxdepth 2 \
        \( -name '*.csv' -o -name '*.log' -o -name config.json -o -name metrics.jsonl \) \
        | tar -czf "$HERE/results2.tgz" -T -
)
say "packed: $HERE/results2.tgz"
say "as delivered: done"
