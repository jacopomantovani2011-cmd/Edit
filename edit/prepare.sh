#!/usr/bin/env bash
# Usage: edit/prepare.sh clip1.mov clip2.mov ... clip7.mov
# Rotates (from phone metadata) and motion-interpolates each clip to 60fps raw frames in work/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work out
i=0
for f in "$@"; do
  i=$((i+1))
  ffmpeg -v error -y -i "$f" \
    -vf "scale=576:1024,minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" \
    -pix_fmt rgb24 -f rawvideo "work/clip$i.rgb" &
done
wait
python3 edit/music.py work/music.wav
echo "Ready: python3 edit/render.py out/edit.mp4"
