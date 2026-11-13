#!/usr/bin/env bash
# The final model: the descriptor head with the two searched trunks (arms nas and nas2, the
# codes runs/nas_final trained), seeds 0-2, on Sentinel-2, Landsat 8, HEO and PeruSat-1 with
# the made-up cameras, as runs/nas_final, plus one change per stage, so that each stage
# differs from the one before it in one thing only:
#   stage 1  --coarser 0.5: half of the batches from Landsat 8, HEO and PeruSat-1 at a pixel
#            2 or 3 times larger, the card told so: RUNS/nas_coarser, packed in results4.tgz
#   stage 2  the same, and --faults 0.5: half of the samples with a calibration fault (near
#            infrared leaking into the visible bands, gains off): RUNS/nas_coarser_faults,
#            packed in results5.tgz
# Each stage is scored when its three seeds are done: Tests A-F (GPU 0), and Tests D-F with
# the scenes predicted on a 10 m and on a 5 m grid (GPU 1). It waits for any training or
# evaluation still running before it starts. Stage 2 is left out when RUNS/final_stage2.skip
# exists when stage 1 ends: another job can claim the GPUs from then on by creating it.
#
#     bash scripts/final.sh        (inside tmux, from ~/projects/catena)
#
# RUNS/final.log says when each step started and how it ended. Started again, it goes on: a
# seed from its last.pt, and a CSV already written is not made again (delete one that a
# failed step left).
set -u
DATA=${DATA:-/root/catena_data}
RUNS=$DATA/runs
HERE=$(cd "$(dirname "$0")/.." && pwd)
LOG=$RUNS/final.log
mkdir -p "$RUNS"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

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

stage() {  # stage NAME PACK [training options]: train nas and nas2, score them, pack OUT
    local out=$RUNS/$1 pack=$HERE/$2
    shift 2
    mkdir -p "$out"
    say "$(basename "$out"): training start (scripts/big.sh --arms nas nas2 $*)"
    OUT="$out" DATA="$DATA" bash scripts/big.sh --arms nas nas2 --trunk $CODES "$@"
    say "$(basename "$out"): training done (see $out/big.log)"
    if finished "$out"; then
        evaluate "$out/tests.csv" "$out/tests.log" "$out" --device cuda:0 &
        {
            evaluate "$out/g10.csv" "$out/g10.log" "$out" --tests D E F --grid 10 \
                --device cuda:1
            evaluate "$out/g5.csv" "$out/g5.log" "$out" --tests D E F --grid 5 --device cuda:1
        } &
        wait
    else
        say "$(basename "$out"): not every seed finished, nothing scored"
    fi
    (
        cd "$RUNS" && {
            echo final.log
            find "$(basename "$out")" -maxdepth 2 \
                \( -name '*.csv' -o -name '*.log' -o -name config.json -o -name metrics.jsonl \)
        } | tar -czf "$pack" -T -
    )
    say "$(basename "$out"): packed in $pack"
}

say "final: start (after any job still running)"
quiet "catena[.]train_clarity" "catena[.]nas" "catena[.]evaluate_clarity" "catena[.]onboard"
TRUNKS="import json, sys; print(' '.join(json.load(open(sys.argv[1]))['args']['trunk']))"
CODES=$(python -c "$TRUNKS" "$RUNS/nas_final/seed0/config.json")
if [ -z "$CODES" ] || [ "$(echo "$CODES" | wc -w)" -ne 2 ]; then
    say "final: no two trunk codes in $RUNS/nas_final/seed0/config.json, nothing trained"
    exit 1
fi
say "final: the trunks of runs/nas_final: $CODES"
stage nas_coarser results4.tgz --coarser 0.5
if [ -f "$RUNS/final_stage2.skip" ]; then
    say "final: stage 2 left out ($RUNS/final_stage2.skip is there)"
else
    stage nas_coarser_faults results5.tgz --coarser 0.5 --faults 0.5
fi
say "final: done"
