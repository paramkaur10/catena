#!/usr/bin/env bash
# Real light: the real test data sorted by how much light it had, the one real-data test of
# what a camera's SNR is worth (low sun and dark ground are where a real camera is noisiest).
# It waits for any training or evaluation still running (scripts/final.sh), then:
#   light.csv    every real test scene of B-E: the sun's zenith, the centre, and the SNR its
#                card states where the noise is known (python -m catena.light, a minute)
#   s2_*.csv     Sentinel-2 as delivered, Test A's real camera, every window of the test split
#                scored on its own by every network (--per-window), with the window's patch and
#                sun zenith: earlier (runs/big, runs/arms, runs/nas_final), delivered,
#                nas_coarser and nas_coarser_faults, those with finished runs, two at a time
#                on the two GPUs; a_*.csv the same windows pooled
#   last         results6.tgz in the project folder
#
#     bash scripts/light.sh        (inside tmux, from ~/projects/catena)
#
# OUT (DATA/runs/light) holds light.log, which says when each step started and how it ended.
# Started again, a CSV already written is not made again (delete one that a failed step left).
set -u
DATA=${DATA:-/root/catena_data}
RUNS=$DATA/runs
OUT=${OUT:-$RUNS/light}
HERE=$(cd "$(dirname "$0")/.." && pwd)
WINDOWS=${WINDOWS:-1000000}
mkdir -p "$OUT"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/light.log"; }

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

trained() {  # trained FOLDER: does it hold a finished run (a final.pt at any depth)?
    [ -n "$(find "$1" -name final.pt -print -quit 2> /dev/null)" ]
}

windows() {  # windows NAME DEVICE FOLDER...: Sentinel-2 as delivered, window by window
    local name=$1 device=$2 folder
    shift 2
    for folder in "$@"; do
        if ! trained "$folder"; then
            say "$name: no finished run in $folder, left out"
            return
        fi
    done
    if [ -f "$OUT/s2_$name.csv" ]; then
        say "$name: $OUT/s2_$name.csv is there already"
        return
    fi
    say "$name: start"
    python -u -m catena.evaluate_clarity "$@" --data "$DATA" --tests A --cameras sentinel2 \
        --windows "$WINDOWS" --per-window "$OUT/s2_$name.csv" --out "$OUT/a_$name.csv" \
        --device "$device" >> "$OUT/$name.log" 2>&1
    say "$name: exit $?"
}

say "light: start (after any job still running)"
quiet "catena[.]train_clarity" "catena[.]nas" "catena[.]evaluate_clarity" "catena[.]onboard"
if [ -f "$OUT/light.csv" ]; then
    say "scenes: $OUT/light.csv is there already"
else
    say "scenes: start"
    python -u -m catena.light "$DATA" --out "$OUT/light.csv" >> "$OUT/scenes.log" 2>&1
    say "scenes: exit $?"
fi
{
    windows earlier cuda:0 "$RUNS/big" "$RUNS/arms" "$RUNS/nas_final"
    windows nas_coarser cuda:0 "$RUNS/nas_coarser"
} &
{
    windows delivered cuda:1 "$RUNS/delivered"
    windows nas_coarser_faults cuda:1 "$RUNS/nas_coarser_faults"
} &
wait
(
    cd "$(dirname "$OUT")" \
        && find "$(basename "$OUT")" -maxdepth 1 \( -name '*.csv' -o -name '*.log' \) \
        | tar -czf "$HERE/results6.tgz" -T -
)
say "packed: $HERE/results6.tgz"
say "light: done"
