#!/usr/bin/env bash
# The architecture search (catena.nas), started again when a fault on the GPU stops it, up to
# 30 times. The search knows the candidate it stopped on (OUT/trying.json): it trains it once
# more, and leaves it out (scored 0) if it stops on it again, so the search goes on.
#
#     bash scripts/search.sh OUT --head RUN/seed0/final.pt [catena.nas options]
#                                                 (inside tmux, from ~/projects/catena)
#
# It ends when the search ends. An error of the search's own (exit 1 or 2) is not started
# again: it would come back the same.
set -u
for TRY in $(seq 1 30); do
    python -u -m catena.nas "$@"
    CODE=$?
    if [ "$CODE" -eq 0 ]; then
        exit 0
    fi
    if [ "$CODE" -ne 3 ] && [ "$CODE" -lt 128 ]; then
        echo "$(date -u +%FT%TZ) catena.nas failed (exit $CODE): not started again" >&2
        exit "$CODE"
    fi
    echo "$(date -u +%FT%TZ) catena.nas stopped (exit $CODE, stop $TRY of 30); again in 15 s" >&2
    sleep 15
done
echo "$(date -u +%FT%TZ) catena.nas stopped 30 times: not started again" >&2
exit 1
