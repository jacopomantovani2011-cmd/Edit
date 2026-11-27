"""Version 2: harder and more cinematic.

- starts on the hit (no slow intro), hard stop at the end that loops back to the start
- cuts on kicks/claps of the funk-phonk groove (135 BPM), more half-beat cuts in drop 2
- cold contrasty grade with bloom on the station lights
- break: slow-mo hero shot, cinema bars, red-only colour pop, silence, then drop 2
- transitions: zoom-through, whip, dip-to-black, glitch slices, stutter frames
- no text
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np

import music_v2 as M
from render import GRAIN, clips, fetch, smooth, speed_at, warp
from config import FPS, H, SRC_FPS, W

BEAT = M.BEAT
TOTAL = M.TOTAL
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAMP = [(0, 2.6), (0.35, 2.3), (0.55, 0.3), (1.0, 0.28)]
VELO = [(0, 2.8), (0.25, 2.4), (0.45, 0.28), (0.82, 0.3), (1.0, 2.4)]


def S(clip, start, end, src, speed=1.25, zoom=(1.05, 1.15), center=(0.5, 0.5), fx=()):
    return dict(clip=clip, start=start, end=end, src=src, speed=speed, zoom=zoom, center=center, fx=set(fx))


SHOTS = [
    # drop 1 - opens straight on the hit
    S(2, 0, 1, 0.3, 1.2, (1.45, 1.1), (0.45, 0.6), ['shake', 'glitch']),
    S(3, 1, 2, 0.3, 1.25, (1.1, 1.22), (0.45, 0.55), ['punch']),
    S(7, 2, 3, 0.3, 1.25, (1.05, 1.18), (0.5, 0.55), ['punch']),
    S(6, 3, 4, 0.4, 1.4, (1.0, 1.1), (0.5, 0.5), ['punch', 'whip']),
    S(5, 4, 6, 0.3, RAMP, (1.0, 1.2), (0.45, 0.55), ['dip']),
    S(2, 6, 7, 1.5, 1.3, (1.12, 1.25), (0.5, 0.55), ['punch']),
    S(3, 7, 8, 1.5, 1.3, (1.05, 1.2), (0.45, 0.55), ['punch', 'zoomout']),
    S(1, 8, 9, 0.4, 1.3, (1.0, 1.1), (0.45, 0.55), ['shake']),
    S(4, 9, 10, 0.4, 1.3, (1.05, 1.18), (0.5, 0.55), ['punch']),
    S(7, 10, 11, 1.4, 1.3, (1.1, 1.2), (0.5, 0.55), ['punch']),
    S(2, 11, 11.5, 2.6, 1.5, (1.25, 1.32), (0.45, 0.55), ['glitch']),
    S(3, 11.5, 12, 2.6, 1.5, (1.25, 1.32), (0.45, 0.55), ['glitch', 'stutter']),
    S(6, 12, 14, 1.6, VELO, (1.0, 1.15), (0.5, 0.5), ['dip', 'whip']),
    S(5, 14, 15, 2.6, 1.3, (1.05, 1.15), (0.45, 0.55), ['punch']),
    S(4, 15, 16, 1.5, 1.3, (1.0, 1.25), (0.5, 0.55), ['punch', 'zoomout']),
    # break - slow-mo hero, cinema bars, red pop, silence
    S(4, 16, 19.5, 2.4, 0.3, (1.25, 1.42), (0.5, 0.5), ['bars', 'redpop']),
    S(4, 19.5, 20, None, 0, (1.42, 1.48), (0.5, 0.5), ['bars', 'redpop', 'freeze', 'dark']),
    # drop 2 - harder, faster
    S(1, 20, 21, 2.3, 1.3, (1.15, 1.0), (0.45, 0.55), ['flash', 'shake', 'glitch']),
    S(2, 21, 22, 3.3, 1.2, (1.05, 1.2), (0.5, 0.55), ['punch']),
    S(3, 22, 22.5, 3.4, 1.4, (1.2, 1.28), (0.45, 0.55), ['punch']),
    S(7, 22.5, 23, 2.6, 1.4, (1.15, 1.22), (0.5, 0.55), ['punch']),
    S(6, 23, 24, 3.1, 1.4, (1.0, 1.12), (0.5, 0.5), ['punch', 'whip']),
    S(5, 24, 26, 3.4, VELO, (1.0, 1.18), (0.45, 0.55), ['shake', 'dip']),
    S(4, 26, 27, 0.2, 1.3, (1.05, 1.2), (0.5, 0.55), ['punch']),
    S(2, 27, 27.5, 3.8, 1.5, (1.3, 1.36), (0.45, 0.6), ['glitch']),
    S(3, 27.5, 28, 3.8, 1.5, (1.3, 1.36), (0.45, 0.55), ['glitch', 'stutter', 'zoomout']),
    S(1, 28, 30, 3.2, VELO, (1.0, 1.2), (0.45, 0.55), ['shake', 'whip']),
    S(7, 30, 31, 3.0, 1.3, (1.1, 1.22), (0.5, 0.55), ['punch']),
    S(6, 31, 31.5, 3.9, 1.5, (1.1, 1.18), (0.5, 0.5), ['punch']),
    S(5, 31.5, 32, 4.2, 1.5, (1.2, 1.3), (0.45, 0.55), ['punch', 'zoomout']),
    # final hit - same framing as the opening so the video loops
    S(2, 32, 32 + M.TAIL / BEAT + 0.05, 0.3, 0.5, (1.5, 1.45), (0.45, 0.6), ['shake', 'flash']),
]

KICKS = np.array(M.kick_times())
CLAPS = np.array(M.clap_times())


def build_time_maps():
    for idx, sh in enumerate(SHOTS):
        t0, t1 = sh['start'] * BEAT, sh['end'] * BEAT
        nfr = max(1, int(round(t1 * FPS)) - int(round(t0 * FPS)))
        sh['t0'], sh['t1'] = t0, t1
        if 'freeze' in sh['fx']:
            sh['srcmap'] = np.full(nfr, SHOTS[idx - 1]['srcmap'][-1])
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
        m = src + cum
        if 'stutter' in sh['fx']:   # hold every frame twice -> choppy step-print look
            m = m[(np.arange(nfr) // 3) * 3]
        sh['srcmap'] = m


# ------------------------------------------------------------------ look

def make_lut():
    x = np.arange(256) / 255.0
    s = np.clip((x - 0.05) / 0.95, 0, 1)
    s = s + 0.18 * np.sin(2 * np.pi * s) * -0.5 * (1 - s)   # deeper shadows
    s = np.clip(0.5 + (s - 0.5) * 1.18, 0, 1)
    r = s * 0.98 + 0.02 * s ** 2
    g = s
    b = np.clip(s + 0.05 * (1 - s) * (s > 0.04) + 0.02, 0, 1)   # cold
    to = lambda c: np.clip(c * 255, 0, 255).astype(np.uint8)
    return to(r), to(g), to(b)


LR, LG, LB = make_lut()
SAT = np.clip(np.arange(256) * 1.12, 0, 255).astype(np.uint8)


def grade(img, redpop=0.0):
    f = img.astype(np.float32)
    # bloom on the station lights / reflections
    lum = f.max(axis=2)
    hi = np.clip(lum - 205, 0, None)[..., None] / 50.0 * f
    small = cv2.resize(hi, (SRC_W_ // 4, SRC_H_ // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6)
    bloom = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    img = np.clip(f + bloom * 0.6, 0, 255).astype(np.uint8)
    r, g, b = cv2.split(img)
    img = cv2.merge([cv2.LUT(r, LR), cv2.LUT(g, LG), cv2.LUT(b, LB)])
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if redpop > 0:
        h = hsv[..., 0].astype(np.int16)
        red = ((h <= 9) | (h >= 168)) & (hsv[..., 1] > 70)
        red = cv2.GaussianBlur(red.astype(np.float32), (0, 0), 2)
        keep = red + (1 - red) * (1 - redpop)
        hsv[..., 1] = np.clip(hsv[..., 1] * keep * 1.25, 0, 255).astype(np.uint8)
    else:
        hsv[..., 1] = cv2.LUT(hsv[..., 1], SAT)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


SRC_H_, SRC_W_ = clips[1].shape[1:3]

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIG = (1 - 0.55 * np.clip((_d - 0.45) / 0.9, 0, 1) ** 1.5)[..., None].astype(np.float32)
del _d, xx, yy


def decay(t, ev, tau):
    p = ev[ev <= t + 1e-6]
    return math.exp(-(t - p[-1]) / tau) if len(p) else 0.0


def shot_at(t):
    for i, sh in enumerate(SHOTS):
        if sh['t0'] - 1e-6 <= t < sh['t1']:
            return i, sh
    return len(SHOTS) - 1, SHOTS[-1]


def glitch(frame, amount, seed):
    r = np.random.default_rng(seed)
    out = frame.copy()
    for _ in range(int(4 + 10 * amount)):
        y0 = r.integers(0, H - 20)
        hgt = int(r.integers(12, 120))
        shift = int(r.normal(0, 80 * amount))
        band = out[y0:y0 + hgt]
        band[:] = np.roll(band, shift, axis=1)
        if r.random() < 0.4:   # channel tear
            band[..., 0] = np.roll(band[..., 0], int(shift * 0.3), axis=1)
    return out


def render_frame(fi):
    t = fi / FPS
    idx, sh = shot_at(t)
    k = max(0, min(int(round(t * FPS)) - int(round(sh['t0'] * FPS)), len(sh['srcmap']) - 1))
    u = (t - sh['t0']) / (sh['t1'] - sh['t0'])
    since, to_end = t - sh['t0'], sh['t1'] - t
    prev = SHOTS[idx - 1] if idx > 0 else None
    fx = sh['fx']
    drums = any(a * BEAT <= t < b * BEAT for a, b in (M.DROP1, M.DROP2))

    redpop = 0.0
    if 'redpop' in fx:
        redpop = smooth((t - 16 * BEAT) / 0.5) * 0.92
    src = grade(fetch(sh['clip'], sh['srcmap'][k]), redpop)

    z0, z1 = sh['zoom']
    zoom = z0 + (z1 - z0) * smooth(u) if 'freeze' not in fx else z0 + (z1 - z0) * u
    cx, cy = sh['center']
    if drums:
        zoom += 0.035 * decay(t, KICKS, 0.07)
    if 'punch' in fx:
        zoom += 0.08 * math.exp(-since / 0.07)
    zt = 0.15
    zo = 'zoomout' in fx and to_end < zt
    land = prev is not None and 'zoomout' in prev['fx'] and since < zt
    if zo:
        zoom *= 1 + 0.9 * smooth(1 - to_end / zt) ** 2
    if land:
        zoom *= 1 + 0.6 * (1 - smooth(since / zt))

    dx = dy = rot = 0.0
    shake = 0.0
    if 'shake' in fx:
        shake = math.exp(-since / 0.2)
    if drums:
        shake = max(shake, 0.18 * decay(t, KICKS, 0.07))
    if 'freeze' in fx:
        shake = 0.2
    if shake > 0.01:
        n = fi * 1.9
        dx = shake * 30 * (math.sin(n * 3.3) + 0.5 * math.sin(n * 7.1))
        dy = shake * 30 * (math.cos(n * 2.9) + 0.5 * math.sin(n * 6.3))
        rot = shake * 1.6 * math.sin(n * 4.1)

    whip = 0.0
    wt = 0.12
    if 'whip' in fx and to_end < wt:
        whip = smooth(1 - to_end / wt)
        dy -= whip * H * 0.3          # vertical whip this time
    if prev is not None and 'whip' in prev['fx'] and since < wt:
        w2 = 1 - smooth(since / wt)
        whip = max(whip, w2)
        dy += w2 * H * 0.3

    frame = warp(src, zoom, cx, cy, dx, dy, rot)

    zb = 0.0
    if zo:
        zb = smooth(1 - to_end / zt)
    if land:
        zb = max(zb, 1 - smooth(since / zt))
    if zb > 0.05:
        acc = frame.astype(np.float32)
        for j in range(1, 6):
            acc += warp(src, zoom * (1 + 0.04 * j * zb), cx, cy, dx, dy, rot)
        frame = (acc / 6).astype(np.uint8)
    if whip > 0.05:
        ks = int(9 + whip * 110) | 1
        kern = np.full((ks, 1), 1.0 / ks, np.float32)
        frame = cv2.filter2D(frame, -1, kern)

    blur = cv2.GaussianBlur(frame, (0, 0), 1.5)
    frame = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)

    if 'glitch' in fx and since < 0.1:
        frame = glitch(frame, 1 - since / 0.1, fi)

    f = frame.astype(np.float32)

    rgb = 0.0
    if drums:
        rgb = 0.6 * decay(t, CLAPS, 0.05)
    rgb = max(rgb, 0.8 * max(zb, whip))
    if 'glitch' in fx:
        rgb = max(rgb, math.exp(-since / 0.1))
    px = int(round(rgb * 14))
    if px:
        f[..., 0] = np.roll(f[..., 0], px, axis=1)
        f[..., 2] = np.roll(f[..., 2], -px, axis=1)

    f *= VIG

    # dip to black on velocity entries, white flash only on the two big hits
    if 'dip' in fx and since < 0.2:
        f *= 0.15 + 0.85 * smooth(since / 0.2)
    if 'flash' in fx and since < 0.25:
        a = math.exp(-since / 0.045)
        f = f * (1 - a) + 255 * a
    if drums:
        f *= 1 + 0.08 * decay(t, CLAPS, 0.05)
    if 'dark' in fx:
        f *= 0.55

    gr = cv2.resize(GRAIN[fi % len(GRAIN)], (W, H), interpolation=cv2.INTER_LINEAR)
    f += gr[..., None] * 3.5
    frame = np.clip(f, 0, 255).astype(np.uint8)

    # animated cinema bars in the break
    if 'bars' in fx:
        a = smooth((t - 16 * BEAT) / 0.35)
        bh = int(a * H * 0.14)
        if bh:
            frame[:bh] = 0
            frame[H - bh:] = 0
    elif prev is not None and 'bars' in prev['fx'] and since < 0.08:
        bh = int((1 - since / 0.08) * H * 0.14)
        if bh:
            frame[:bh] = 0
            frame[H - bh:] = 0
    return frame


def main(out_path, preview=False):
    audio = os.path.join(ROOT, 'work', 'music_v2.wav')
    M.build(audio)
    build_time_maps()
    nframes = int(TOTAL * FPS)
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-i', audio,
           '-c:v', 'libx264', '-preset', 'veryfast' if preview else 'slow',
           '-crf', '23' if preview else '19', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
           '-c:a', 'aac', '-b:a', '256k', '-shortest', '-movflags', '+faststart', out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(nframes):
        p.stdin.write(render_frame(fi).tobytes())
    p.stdin.close()
    p.wait()
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'out', 'edit_v2.mp4'),
         preview='--preview' in sys.argv)
