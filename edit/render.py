"""Beat-synced TikTok-style edit renderer.

Reads motion-interpolated 60fps raw clips, applies a timeline of shots with
speed ramps, punch zooms, shake, flashes, RGB split, zoom/whip transitions,
a night colour grade, grain and animated titles, and pipes frames to ffmpeg.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (BEAT, FPS, H, INTRO_END, OUTRO, PART_B, SONG_END, SRC_FPS, SRC_H,
                    SRC_W, TOTAL, W, kick_times, snare_times)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, 'work')
FONT = os.path.join(ROOT, 'assets', 'fonts', 'anton-latin-400-normal.woff')

# ------------------------------------------------------------------ sources

clips = {}
for i in range(1, 8):
    p = os.path.join(WORK, f'clip{i}.rgb')
    n = os.path.getsize(p) // (SRC_W * SRC_H * 3)
    clips[i] = np.memmap(p, np.uint8, 'r', shape=(n, SRC_H, SRC_W, 3))

# Shot list. Times are in beats. Fields:
#   clip, start, end, src (in-point seconds), speed (number or [(u, speed), ...]),
#   zoom (z0, z1), center (cx, cy) normalised focus point,
#   fx: set of 'flash', 'punch', 'shake', 'rgb', 'zoomout' (transition into next),
#       'whip' (whip-pan out), 'freeze', 'fadein'
RAMP = [(0, 2.6), (0.32, 2.4), (0.5, 0.35), (1.0, 0.3)]          # fast -> slow-mo
VELO = [(0, 2.8), (0.22, 2.4), (0.42, 0.3), (0.8, 0.3), (1.0, 2.2)]  # fast-slow-fast

SHOTS = [
    # ---- intro: moody slow-mo, muffled beat
    dict(clip=1, start=0, end=4, src=0.2, speed=0.55, zoom=(1.0, 1.14), center=(0.45, 0.5), fx={'fadein'}),
    dict(clip=4, start=4, end=7.5, src=0.6, speed=[(0, 0.45), (0.6, 0.6), (1.0, 1.9)],
         zoom=(1.05, 1.35), center=(0.5, 0.48), fx=set()),
    dict(clip=4, start=7.5, end=8, src=None, speed=0, zoom=(1.35, 1.42), center=(0.5, 0.48), fx={'freeze'}),
    # ---- drop: hard cuts on the beat
    dict(clip=5, start=8, end=9, src=0.3, speed=1.3, zoom=(1.12, 1.0), center=(0.5, 0.55), fx={'flash', 'shake', 'rgb'}),
    dict(clip=2, start=9, end=10, src=0.2, speed=1.2, zoom=(1.1, 1.18), center=(0.5, 0.5), fx={'punch'}),
    dict(clip=3, start=10, end=11, src=0.2, speed=1.2, zoom=(1.05, 1.2), center=(0.45, 0.55), fx={'punch'}),
    dict(clip=6, start=11, end=12, src=0.3, speed=1.4, zoom=(1.0, 1.08), center=(0.5, 0.5), fx={'punch', 'rgb'}),
    dict(clip=1, start=12, end=14, src=1.6, speed=RAMP, zoom=(1.0, 1.2), center=(0.45, 0.55), fx={'flash', 'shake'}),
    dict(clip=7, start=14, end=14.5, src=0.2, speed=1.4, zoom=(1.1, 1.15), center=(0.5, 0.5), fx={'punch'}),
    dict(clip=2, start=14.5, end=15, src=1.6, speed=1.4, zoom=(1.2, 1.3), center=(0.45, 0.6), fx={'punch'}),
    dict(clip=4, start=15, end=16, src=2.4, speed=1.2, zoom=(1.0, 1.25), center=(0.5, 0.55), fx={'punch', 'zoomout'}),
    dict(clip=3, start=16, end=17, src=1.6, speed=1.2, zoom=(1.15, 1.0), center=(0.45, 0.55), fx={'flash', 'shake'}),
    dict(clip=6, start=17, end=18, src=1.4, speed=1.3, zoom=(1.0, 1.1), center=(0.5, 0.45), fx={'punch'}),
    dict(clip=5, start=18, end=19, src=1.5, speed=1.3, zoom=(1.0, 1.12), center=(0.45, 0.55), fx={'punch', 'rgb'}),
    dict(clip=7, start=19, end=20, src=1.2, speed=1.2, zoom=(1.0, 1.15), center=(0.5, 0.55), fx={'punch', 'whip'}),
    dict(clip=4, start=20, end=22, src=0.2, speed=RAMP, zoom=(1.0, 1.18), center=(0.5, 0.55), fx={'flash', 'shake', 'rgb'}),
    dict(clip=2, start=22, end=22.5, src=2.6, speed=1.5, zoom=(1.15, 1.2), center=(0.5, 0.5), fx={'flash'}),
    dict(clip=3, start=22.5, end=23, src=2.8, speed=1.5, zoom=(1.2, 1.25), center=(0.45, 0.5), fx={'flash'}),
    dict(clip=6, start=23, end=23.5, src=2.4, speed=1.5, zoom=(1.1, 1.15), center=(0.5, 0.5), fx={'flash'}),
    dict(clip=5, start=23.5, end=24, src=2.6, speed=1.5, zoom=(1.2, 1.3), center=(0.5, 0.55), fx={'flash', 'zoomout'}),
    # ---- part B: velocity ramps, 2 beats each
    dict(clip=6, start=24, end=26, src=2.9, speed=VELO, zoom=(1.0, 1.15), center=(0.5, 0.5), fx={'flash', 'shake', 'whip'}),
    dict(clip=5, start=26, end=28, src=3.3, speed=VELO, zoom=(1.0, 1.15), center=(0.45, 0.55), fx={'shake', 'whip'}),
    dict(clip=2, start=28, end=30, src=2.9, speed=VELO, zoom=(1.0, 1.2), center=(0.5, 0.55), fx={'shake', 'whip'}),
    dict(clip=7, start=30, end=32, src=2.2, speed=VELO, zoom=(1.0, 1.15), center=(0.5, 0.5), fx={'shake', 'zoomout'}),
    # ---- outro: hero slow-mo
    dict(clip=4, start=32, end=SONG_END + 1.2 / BEAT, src=3.0, speed=0.32, zoom=(1.18, 1.0), center=(0.5, 0.5),
         fx={'flash', 'shake', 'rgb'}),
]

TITLES = [
    # text, start beat, end beat, y (fraction of height), size
    ('NIGHT RIDE', 1.0, 3.6, 0.40, 165),
    ('2 TEMPI', 32.5, SONG_END + 1.0, 0.40, 190),
]

KICKS = np.array(kick_times())
SNARES = np.array(snare_times())

# ---------------------------------------------------------------- utilities


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def speed_at(spec, u):
    if not isinstance(spec, list):
        return float(spec)
    for (u0, s0), (u1, s1) in zip(spec, spec[1:]):
        if u0 <= u <= u1:
            k = (u - u0) / max(u1 - u0, 1e-6)
            return s0 + (s1 - s0) * smooth(k)
    return spec[-1][1]


def build_time_maps():
    """For each shot, precompute output-frame -> source-time mapping (integrated speed)."""
    for idx, sh in enumerate(SHOTS):
        t0, t1 = sh['start'] * BEAT, sh['end'] * BEAT
        nfr = max(1, int(round(t1 * FPS)) - int(round(t0 * FPS)))
        sh['t0'], sh['t1'] = t0, t1
        if 'freeze' in sh['fx']:
            prev = SHOTS[idx - 1]
            sh['srcmap'] = np.full(nfr, prev['srcmap'][-1])
            continue
        sub = 8
        dt = (t1 - t0) / (nfr * sub)
        us = (np.arange(nfr * sub) + 0.5) / (nfr * sub)
        sp = np.array([speed_at(sh['speed'], u) for u in us])
        cum = np.concatenate([[0], np.cumsum(sp * dt)])[::sub][:nfr]
        dur = (len(clips[sh['clip']]) - 2) / SRC_FPS
        total = cum[-1] + sp[-sub:].mean() * dt * sub
        src = sh['src']
        if src + total > dur:
            src = max(0.0, dur - total)
            print(f'  shot {idx} clip{sh["clip"]}: in-point clamped to {src:.2f}s', file=sys.stderr)
        sh['srcmap'] = src + cum


def fetch(clip, t):
    arr = clips[clip]
    f = t * SRC_FPS
    i = int(math.floor(f))
    i = min(max(i, 0), len(arr) - 2)
    a = f - i
    if a < 0.15:
        return np.asarray(arr[i])
    if a > 0.85:
        return np.asarray(arr[i + 1])
    return cv2.addWeighted(np.asarray(arr[i]), 1 - a, np.asarray(arr[i + 1]), a, 0)


# --------------------------------------------------------------- colour grade

def make_luts():
    x = np.arange(256) / 255.0
    # S-curve with crushed blacks and soft highlight roll-off
    s = np.clip((x - 0.04) / 0.96, 0, 1)
    s = 0.5 - 0.5 * np.cos(np.pi * s) * 0.55 + (s - 0.5) * 0.45
    s = np.clip(s, 0, 1) ** 1.08
    r = s + 0.035 * np.sin(np.pi * s) * (s > 0.5)          # warm highs
    g = s + 0.010 * np.sin(np.pi * s)
    b = s + 0.06 * (1 - s) ** 2 * (s < 0.5) - 0.02 * s    # teal-ish shadows
    to = lambda c: np.clip(c * 255, 0, 255).astype(np.uint8)
    return to(r), to(g), to(b)


LUT_R, LUT_G, LUT_B = make_luts()


def grade(img):
    r, g, b = cv2.split(img)
    img = cv2.merge([cv2.LUT(r, LUT_R), cv2.LUT(g, LUT_G), cv2.LUT(b, LUT_B)])
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = cv2.LUT(hsv[..., 1], SAT_LUT)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


SAT_LUT = np.clip(np.arange(256) * 1.28, 0, 255).astype(np.uint8)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIGNETTE = (1 - 0.42 * np.clip((_d - 0.55) / 0.85, 0, 1) ** 1.6)[..., None].astype(np.float32)
del _d

rng = np.random.default_rng(3)
GRAIN = [rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]

# ---------------------------------------------------------------- titles


def render_title(text, size):
    font = ImageFont.truetype(FONT, size)
    spacing = int(size * 0.06)
    widths = [font.getbbox(ch)[2] for ch in text]
    tw = sum(widths) + spacing * (len(text) - 1)
    pad = 60
    img = Image.new('RGBA', (tw + pad * 2, int(size * 1.6) + pad * 2), (0, 0, 0, 0))
    glow = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d, dg = ImageDraw.Draw(img), ImageDraw.Draw(glow)
    x = pad
    for ch, w in zip(text, widths):
        dg.text((x, pad), ch, font=font, fill=(255, 40, 40, 255))
        d.text((x, pad), ch, font=font, fill=(255, 255, 255, 255))
        x += w + spacing
    g = np.array(glow).astype(np.float32)
    g[..., 3] = cv2.GaussianBlur(g[..., 3], (0, 0), size * 0.12) * 0.9
    base = np.array(img).astype(np.float32)
    # composite white text over red glow (premultiplied-ish)
    a_t = base[..., 3:4] / 255
    a_g = g[..., 3:4] / 255
    rgb = base[..., :3] * a_t + g[..., :3] * a_g * (1 - a_t)
    a = a_t + a_g * (1 - a_t)
    out = np.dstack([rgb / np.maximum(a, 1e-4), a * 255]).astype(np.float32)
    return out


TITLE_IMGS = [render_title(t, s) for t, _, _, _, s in TITLES]


def overlay(frame, rgba, cx, cy, scale, alpha, shift=0):
    if alpha <= 0.01:
        return frame
    h, w = rgba.shape[:2]
    nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
    im = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + nw), min(H, y0 + nh)
    if xb <= xa or yb <= ya:
        return frame
    sub = im[ya - y0:yb - y0, xa - x0:xb - x0]
    a = sub[..., 3:4] / 255 * alpha
    roi = frame[ya:yb, xa:xb].astype(np.float32)
    col = sub[..., :3]
    if shift:  # chromatic split on the title
        col = col.copy()
        col[..., 0] = np.roll(col[..., 0], shift, axis=1)
        col[..., 2] = np.roll(col[..., 2], -shift, axis=1)
    frame[ya:yb, xa:xb] = (roi * (1 - a) + col * a).astype(np.uint8)
    return frame


# ---------------------------------------------------------------- per frame

def warp(src, zoom, cx, cy, dx=0.0, dy=0.0, rot=0.0):
    s = W / SRC_W * zoom
    # keep the crop inside the source
    half_w, half_h = W / (2 * s), H / (2 * s)
    px = np.clip(cx * SRC_W, half_w, SRC_W - half_w)
    py = np.clip(cy * SRC_H, half_h, SRC_H - half_h)
    M = cv2.getRotationMatrix2D((float(px), float(py)), rot, s)
    M[0, 2] += W / 2 - px + dx
    M[1, 2] += H / 2 - py + dy
    return cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def decay_since(t, events, tau):
    past = events[events <= t + 1e-6]
    if len(past) == 0:
        return 0.0
    return math.exp(-(t - past[-1]) / tau)


def shot_at(t):
    for i, sh in enumerate(SHOTS):
        if sh['t0'] - 1e-6 <= t < sh['t1']:
            return i, sh
    return len(SHOTS) - 1, SHOTS[-1]


def render_frame(fi):
    t = fi / FPS
    idx, sh = shot_at(t)
    k = min(int(round(t * FPS)) - int(round(sh['t0'] * FPS)), len(sh['srcmap']) - 1)
    k = max(k, 0)
    u = (t - sh['t0']) / (sh['t1'] - sh['t0'])
    src = grade(fetch(sh['clip'], sh['srcmap'][k]))

    z0, z1 = sh['zoom']
    zoom = z0 + (z1 - z0) * (u if 'freeze' not in sh['fx'] else u)
    cx, cy = sh['center']
    in_drums = INTRO_END * BEAT <= t < OUTRO * BEAT
    since_cut = t - sh['t0']
    to_end = sh['t1'] - t

    # beat bounce on every kick
    if in_drums:
        zoom += 0.045 * decay_since(t, KICKS, 0.07)
    if 'punch' in sh['fx']:
        zoom += 0.10 * math.exp(-since_cut / 0.06)
    # zoom-in transition out of a shot / landing zoom-out into the next
    if 'zoomout' in sh['fx'] and to_end < 0.17:
        zoom *= 1 + 0.8 * smooth(1 - to_end / 0.17) ** 2
    prev = SHOTS[idx - 1] if idx > 0 else None
    land = prev is not None and 'zoomout' in prev['fx'] and since_cut < 0.17
    if land:
        zoom *= 1 + 0.5 * (1 - smooth(since_cut / 0.17))

    dx = dy = rot = 0.0
    shake = 0.0
    if 'shake' in sh['fx']:
        shake = max(shake, math.exp(-since_cut / 0.18))
    if 'freeze' in sh['fx']:
        shake = max(shake, 0.35)
    if in_drums:
        shake = max(shake, 0.25 * decay_since(t, KICKS, 0.08))
    if shake > 0.01:
        n = fi * 1.7
        dx = shake * 28 * (math.sin(n * 3.1) + 0.5 * math.sin(n * 7.3))
        dy = shake * 28 * (math.cos(n * 2.7) + 0.5 * math.sin(n * 5.9))
        rot = shake * 1.4 * math.sin(n * 4.3)

    # whip pan out of a shot / in to the next
    whip = 0.0
    if 'whip' in sh['fx'] and to_end < 0.13:
        whip = smooth(1 - to_end / 0.13)
        dx += whip * W * 0.35
    if prev is not None and 'whip' in prev['fx'] and since_cut < 0.13:
        w2 = 1 - smooth(since_cut / 0.13)
        whip = max(whip, w2)
        dx -= w2 * W * 0.35

    frame = warp(src, zoom, cx, cy, dx, dy, rot)

    # motion blur for transitions
    zoom_blur = 0.0
    if 'zoomout' in sh['fx'] and to_end < 0.17:
        zoom_blur = smooth(1 - to_end / 0.17)
    if land:
        zoom_blur = max(zoom_blur, 1 - smooth(since_cut / 0.17))
    if zoom_blur > 0.05:
        acc = frame.astype(np.float32)
        steps = 6
        for j in range(1, steps):
            f2 = warp(src, zoom * (1 + 0.035 * j * zoom_blur), cx, cy, dx, dy, rot)
            acc += f2
        frame = (acc / steps).astype(np.uint8)
    if whip > 0.05:
        ks = int(9 + whip * 90) | 1
        kern = np.zeros((1, ks), np.float32)
        kern[0, :] = 1.0 / ks
        frame = cv2.filter2D(frame, -1, kern)

    if 'freeze' in sh['fx']:   # desaturated freeze-frame before the drop
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.addWeighted(frame, 0.35, cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), 0.65, -10)

    # sharpen (sources are upscaled ~1.9x)
    blur = cv2.GaussianBlur(frame, (0, 0), 1.6)
    frame = cv2.addWeighted(frame, 1.55, blur, -0.55, 0)

    f = frame.astype(np.float32)

    # RGB split on hits
    rgb_amt = 0.0
    if 'rgb' in sh['fx']:
        rgb_amt = math.exp(-since_cut / 0.12)
    if in_drums:
        rgb_amt = max(rgb_amt, 0.5 * decay_since(t, SNARES, 0.06))
    if zoom_blur > 0.05 or whip > 0.05:
        rgb_amt = max(rgb_amt, 0.7 * max(zoom_blur, whip))
    sh_px = int(round(rgb_amt * 16))
    if sh_px:
        f[..., 0] = np.roll(f[..., 0], sh_px, axis=1)
        f[..., 2] = np.roll(f[..., 2], -sh_px, axis=1)

    f *= VIGNETTE

    # flashes
    flash = 0.0
    if 'flash' in sh['fx']:
        flash = math.exp(-since_cut / 0.05)
    if in_drums:
        flash = max(flash, 0.12 * decay_since(t, SNARES, 0.05))
    if flash > 0.01:
        f = f * (1 - flash) + 255 * flash

    # fade in / out
    if 'fadein' in sh['fx'] and since_cut < 0.5:
        f *= smooth(since_cut / 0.5)
    end_fade = TOTAL - t
    if end_fade < 0.7:
        f *= smooth(end_fade / 0.7)

    # film grain
    gr = cv2.resize(GRAIN[fi % len(GRAIN)], (W, H), interpolation=cv2.INTER_LINEAR)
    f += gr[..., None] * 4.0

    frame = np.clip(f, 0, 255).astype(np.uint8)

    # titles
    for (text, b0, b1, ty, _), img in zip(TITLES, TITLE_IMGS):
        t0, t1 = b0 * BEAT, b1 * BEAT
        if t0 <= t < t1:
            a_in = smooth((t - t0) / 0.12)
            a_out = smooth((t1 - t) / 0.2)
            scale = 1.0 + 0.35 * (1 - smooth((t - t0) / 0.18)) + 0.03 * (t - t0)
            jitter = int(round(6 * math.exp(-(t - t0) / 0.15)))
            frame = overlay(frame, img, W / 2, H * ty, scale, min(a_in, a_out), jitter)
    return frame


def main(out_path, preview=False):
    build_time_maps()
    nframes = int(TOTAL * FPS)
    audio = os.path.join(WORK, 'music.wav')
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-i', audio,
           '-c:v', 'libx264', '-preset', 'veryfast' if preview else 'slow',
           '-crf', '23' if preview else '19', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
           '-c:a', 'aac', '-b:a', '256k', '-shortest', '-movflags', '+faststart', out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(nframes):
        p.stdin.write(render_frame(fi).tobytes())
        if fi % 60 == 0:
            print(f'frame {fi}/{nframes}', file=sys.stderr)
    p.stdin.close()
    p.wait()
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'out', 'edit.mp4'),
         preview='--preview' in sys.argv)
