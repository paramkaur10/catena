#!/usr/bin/env bash
# The long run, days without a hand on it. It waits for the jobs before it (the arms in
# training, searches 0 and 1, the evaluations of runs/big), then:
#   GPU 0  architecture search long0 (seed 10), and beside it: the arms scored on Tests A-F,
#          Clarity (runs/big) on a 10 m grid (Tests D-F) and in 8 bits (Tests A-F);
#   GPU 1  architecture search long1 (seed 11), and beside it the card checks of the arms that
#          read the card (Tests A and C);
#   both   the two best trunks of long0 and long1 trained end to end like Clarity (arms nas and
#          nas2, seeds 0-2, scripts/big.sh), then scored on Tests A-F, card-checked on A and C,
#          on a 10 m grid and in 8 bits;
#   last   what flies (catena.onboard): sizes, multiply-adds and speeds of every arm, timed with
#          nothing else running.
#
#     bash scripts/long.sh                        (inside tmux, from ~/projects/catena)
#     POPULATION=20 GENERATIONS=10 STEPS=3000 bash scripts/long.sh     (the defaults)
#
# OUT (DATA/runs/long) holds long.log, which says when each stage started and how it ended,
# and each evaluation's log; each search logs to runs/nas/<name>.log. A search that faults on
# the GPU trains on at the level of caution that works (catena.nas), or gives up within
# minutes if none does; the rest of the run goes on either way. Started again, it goes on: a
# search from its last generation, the training from its seeds' last.pt, and an evaluation
# whose CSV is already written is not run again.
set -u
DATA=${DATA:-/root/catena_data}
RUNS=$DATA/runs
OUT=${OUT:-$RUNS/long}
HEAD=${HEAD:-$RUNS/big/seed0/final.pt}
POPULATION=${POPULATION:-20}
GENERATIONS=${GENERATIONS:-10}
STEPS=${STEPS:-3000}
FINAL=$RUNS/nas_final
BIG=$RUNS/big
CARDS=(honest shuffled "pixel x3" "pixel /3" "wavelengths only" "rgbn only" "rgb only")
mkdir -p "$OUT" "$RUNS/nas"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/long.log"; }

quiet() {  # wait until no process matches any pattern, twice in a row a minute apart
    local free=0 busy p
    while [ "$free" -lt 2 ]; do
        busy=0
        for p in "$@"; do
            if pgrep -f "$p" > /dev/null; then busy=1; fi
        done
        if [ "$busy" -eq 1 ]; then free=0; else free=$((free + 1)); fi
        if [ "$free" -lt 2 ]; then sleep 60; fi
    done
}

search() {  # search NAME DEVICE SEED
    local code
    say "$1: start on $2, seed $3, $POPULATION x $GENERATIONS, $STEPS steps"
    RESTARTS=400 bash scripts/search.sh "$RUNS/nas/$1" --head "$HEAD" \
        --population "$POPULATION" --generations "$GENERATIONS" --steps "$STEPS" \
        --k-best 2 --n-random 2 --workers 4 --device "$2" --seed "$3" >> "$RUNS/nas/$1.log" 2>&1
    code=$?
    if [ "$code" -eq 5 ]; then
        say "$1: gave up, every candidate took the process down (see $RUNS/nas/$1.log)"
    else
        say "$1: exit $code"
    fi
}

evaluate() {  # evaluate CSV LOG [catena.evaluate_clarity arguments]
    local csv=$1 log=$2
    shift 2
    if [ -f "$csv" ]; then
        say "$(basename "$log" .log): $csv is there already"
        return
    fi
    say "$(basename "$log" .log): start"
    python -u -m catena.evaluate_clarity "$@" >> "$log" 2>&1
    say "$(basename "$log" .log): exit $?"
}

gpu0() {
    quiet "big[.]sh --arms" "search[.]sh $RUNS/nas/search0" "catena[.]nas $RUNS/nas/search0"
    {
        evaluate "$RUNS/arms/clarity_tests.csv" "$OUT/eval_arms.log" "$RUNS/arms" --device cuda:0
        evaluate "$BIG/clarity_tests_DEF_grid10.csv" "$OUT/grid_big.log" "$BIG" --tests D E F \
            --grid 10 --device cuda:0
        evaluate "$BIG/clarity_tests_ABCDEF_int8.csv" "$OUT/int8_big.log" "$BIG" --int8 \
            --device cuda:0
    } &
    search long0 cuda:0 10
    wait
}

gpu1() {
    quiet "big[.]sh --arms" "search[.]sh $RUNS/nas/search1" "catena[.]nas $RUNS/nas/search1" \
        "catena[.]evaluate_clarity $RUNS/big"
    evaluate "$RUNS/arms/chain/clarity_tests_AC_cards.csv" "$OUT/cards_arms.log" \
        "$RUNS/arms/chain" --tests A C --windows 200 --device cuda:1 \
        --cards "${CARDS[@]}" "noise x3" "blur x2" &
    search long1 cuda:1 11
    wait
}

say "long run: waiting for the jobs before it"
gpu0 &
gpu1 &
wait

TOP=$(python -c "import sys; from catena import nas; print(' '.join(nas.top_codes(sys.argv[1:])))" \
    "$RUNS/nas/long0/search.jsonl" "$RUNS/nas/long1/search.jsonl")
set -- $TOP
TIMED=("$BIG" "$RUNS/arms")
if [ $# -lt 2 ]; then
    say "fewer than two trunks found ($TOP): none trained"
else
    say "the two best trunks: $1 $2"
    say "nas_final: start (scripts/big.sh, seeds 0-2)"
    OUT=$FINAL bash scripts/big.sh --arms nas nas2 --trunk "$1" "$2"
    say "nas_final: done (see $FINAL/big.log)"
    {
        evaluate "$FINAL/clarity_tests_AC_cards.csv" "$OUT/cards_nas.log" "$FINAL" --tests A C \
            --windows 200 --device cuda:1 --cards "${CARDS[@]}"
        evaluate "$FINAL/clarity_tests_DEF_grid10.csv" "$OUT/grid_nas.log" "$FINAL" \
            --tests D E F --grid 10 --device cuda:1
    } &
    evaluate "$FINAL/clarity_tests.csv" "$OUT/eval_nas.log" "$FINAL" --device cuda:0
    evaluate "$FINAL/clarity_tests_ABCDEF_int8.csv" "$OUT/int8_nas.log" "$FINAL" --int8 \
        --device cuda:0
    wait
    TIMED+=("$FINAL")
fi
if [ -f "$OUT/onboard.csv" ]; then
    say "onboard: $OUT/onboard.csv is there already"
else
    say "onboard: start (with nothing else running)"
    python -u -m catena.onboard "${TIMED[@]}" --device cuda:0 --out "$OUT/onboard.csv" \
        >> "$OUT/onboard.log" 2>&1
    say "onboard: exit $?"
fi
say "long run: done"
