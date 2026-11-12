#!/usr/bin/env bash
# The night after scripts/delivered.sh: the working pixel on the satellites never seen, and a
# run that learns from calibration faults. It waits for any training or evaluation still
# running, then:
#   both    arms sim and sensei, seeds 0-2 (scripts/big.sh), on Sentinel-2, Landsat 8 and
#           PeruSat-1 (no HEO), half of sim's samples with a calibration fault (--faults 0.5:
#           near infrared leaking into the visible bands, gains off band by band): FAULTS
#   beside  the scenes of Tests D-F predicted on grids of 5, 10, 15, 20 and 30 m (--grid) and
#           scored on their own pixels, one CSV per grid:
#           GPU 0  every arm of every seed so far (runs/big, runs/arms, runs/nas_final): all_g*
#           GPU 1  the arms of runs/delivered, if its three seeds finished: delivered_g*
#   then    GPU 0: sim and sensei with faults on Tests A-F: FAULTS/tests.csv
#           GPU 1: the two on Tests D-F on a 10 m grid: FAULTS/g10.csv
#   last    results3.tgz in the project folder: every CSV, log, config and metrics of OUT and
#           FAULTS
#
#     bash scripts/night.sh        (inside tmux, from ~/projects/catena)
#
# OUT (DATA/runs/night) holds night.log, which says when each step started and how it ended,
# and the grids' CSVs and logs; FAULTS (DATA/runs/faults) each seed's folder. Started again,
# it goes on: a seed from its last.pt, and a CSV already written is not made again (delete
# one that a failed step left).
set -u
DATA=${DATA:-/root/catena_data}
RUNS=$DATA/runs
OUT=${OUT:-$RUNS/night}
FAULTS=${FAULTS:-$RUNS/faults}
GRIDS=${GRIDS:-5 10 15 20 30}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT" "$FAULTS"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/night.log"; }

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

grids() {  # grids NAME DEVICE RUNS...: the scenes of Tests D-F on each grid of GRIDS
    local name=$1 device=$2 g
    shift 2
    for g in $GRIDS; do
        evaluate "$OUT/${name}_g$g.csv" "$OUT/${name}_g$g.log" "$@" --tests D E F --grid "$g" \
            --device "$device"
    done
}

say "night: start (after any job still running)"
quiet "catena[.]train_clarity" "catena[.]nas" "catena[.]evaluate_clarity" "catena[.]onboard"
grids all cuda:0 "$RUNS/big" "$RUNS/arms" "$RUNS/nas_final" &
if finished "$RUNS/delivered"; then
    grids delivered cuda:1 "$RUNS/delivered" &
else
    say "runs/delivered: not every seed finished, its grids are left out"
fi
say "training: start (scripts/big.sh: sim and sensei with calibration faults, seeds 0-2, no HEO)"
OUT="$FAULTS" DATA="$DATA" bash scripts/big.sh --arms sim sensei --faults 0.5 \
    --shares 0.5 0.25 0 0.25
say "training: done (see $FAULTS/big.log)"
wait

if finished "$FAULTS"; then
    evaluate "$FAULTS/tests.csv" "$FAULTS/tests.log" "$FAULTS" --device cuda:0 &
    evaluate "$FAULTS/g10.csv" "$FAULTS/g10.log" "$FAULTS" --tests D E F --grid 10 \
        --device cuda:1 &
    wait
else
    say "not every seed finished: nothing scored (see $FAULTS/big.log and the seeds' train.log)"
fi

for FOLDER in "$OUT" "$FAULTS"; do
    (
        cd "$(dirname "$FOLDER")" && find "$(basename "$FOLDER")" -maxdepth 2 \
            \( -name '*.csv' -o -name '*.log' -o -name config.json -o -name metrics.jsonl \)
    )
done > "$OUT/packed.txt"
if [ "$(dirname "$OUT")" = "$(dirname "$FAULTS")" ]; then
    tar -czf "$HERE/results3.tgz" -C "$(dirname "$OUT")" -T "$OUT/packed.txt"
    say "packed: $HERE/results3.tgz"
else
    say "OUT and FAULTS are not in one folder: nothing packed"
fi
say "night: done"
