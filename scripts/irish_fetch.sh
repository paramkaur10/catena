#!/usr/bin/env bash
# L7 Irish (Landsat 7 ETM+, 206 scenes in 9 latitude zones) from TorchGeo's mirror on Hugging
# Face, pinned to one revision and checked against TorchGeo's MD5 sums. Each zone's archive is
# downloaded, checked, unpacked into OUT/<zone>/ and deleted; a zone already unpacked is
# skipped, so the script can be started again after a stop. The smallest zone comes first.
#
#     bash scripts/irish_fetch.sh            (inside tmux, from ~/projects/catena)
set -u
OUT=${OUT:-/root/catena_data/chips/l7_irish}
TMP=${TMP:-/root/catena_data/raw/l7_irish}
URL=https://hf.co/datasets/torchgeo/l7irish/resolve/6807e0b22eca7f9a8a3903ea673b31a115837464
NEED_GB=70  # the unpacked scenes (about 55 GB) and one archive at a time
declare -A MD5=(
    [polar_south]=c1503db1cf46d5c37b579190f989e7ec
    [austral]=0a34770b992a62abeb88819feb192436
    [boreal]=b7cfdd689a3c2fd2a8d572e1c10ed082
    [mid_latitude_north]=c40abe5ad2487f8ab021cfb954982faa
    [mid_latitude_south]=37abab7f6ebe3d6cf6a3332144145427
    [polar_north]=49d9e616bd715057db9acb1c4d234d45
    [subtropical_north]=a6010de4c50167260de35beead9d6a65
    [subtropical_south]=c37d439df2f05bd7cfe87cf6ff61a690
    [tropical]=d7931419c70f3520a17361d96f1a4810
)
ZONES="polar_south austral polar_north tropical subtropical_north mid_latitude_south"
ZONES="$ZONES mid_latitude_north boreal subtropical_south"
mkdir -p "$OUT" "$TMP"
FREE=$(df --output=avail -BG "$OUT" | tail -n 1 | tr -dc 0-9)
if [ "$FREE" -lt "$NEED_GB" ]; then
    echo "only ${FREE} GB free under $OUT; ${NEED_GB} GB needed"
    exit 1
fi
for ZONE in $ZONES; do
    if [ -f "$OUT/$ZONE/.done" ]; then
        echo "$(date -u +%FT%TZ) $ZONE: already unpacked"
        continue
    fi
    FILE="$TMP/$ZONE.tar.gz"
    echo "$(date -u +%FT%TZ) $ZONE: downloading"
    if ! curl -L --fail --retry 5 --silent --show-error -C - -o "$FILE" "$URL/$ZONE.tar.gz"; then
        echo "$(date -u +%FT%TZ) $ZONE: download failed"
        continue
    fi
    if [ "$(md5sum < "$FILE" | cut -c1-32)" != "${MD5[$ZONE]}" ]; then
        echo "$(date -u +%FT%TZ) $ZONE: MD5 does not match; the archive stays in $TMP"
        continue
    fi
    mkdir -p "$OUT/$ZONE"
    if tar -xzf "$FILE" -C "$OUT/$ZONE"; then
        touch "$OUT/$ZONE/.done"
        rm -f "$FILE"
        echo "$(date -u +%FT%TZ) $ZONE: unpacked, $(du -sh "$OUT/$ZONE" | cut -f1)"
    else
        echo "$(date -u +%FT%TZ) $ZONE: unpacking failed"
    fi
done
echo "$(date -u +%FT%TZ) done"
