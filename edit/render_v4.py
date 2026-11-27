"""Version 4: the current TikTok moto trend stack on top of the v3 structure:
glitch transitions + low-saturation cold contrasty grade + beat-synced camera moves
(step zooms that snap on every beat), scale bounce 100->115->100 on the hits,
slowest point of each velocity ramp exactly on a beat.

v3 notes:

- hook: motion from frame 1; build = 2 long holds (4 beats each), filtered music
- strongest shot lands exactly on the drop; then 2-beat alternation; then a
  1-beat burst of WIDE, readable shots; strong slow-mo closing shot
- speed ramps only on 3 shots (drop, mid, ending): approach -> slow hold -> exit
- shake (amplitude decaying over ~20 frames, motion-blurred) + flash ONLY on the
  3 big hits (drop, burst, final hit); other cuts are clean hard cuts
- one transition per section change (zoom into the drop, whip into the burst)
- real motion blur from frame blending when footage runs faster than real time
- clean grade: contrast up, highlights down, shadows slightly lifted, paint pops
- 60 fps export, 1080x1920, high bitrate
"""
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np

import music_v3 as M
from render import clips, smooth, speed_at, warp
from config import H, SRC_FPS, W

FPS = 60
BEAT = M.BEAT
TOTAL = M.TOTAL
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# approach -> slow hold -> exit (smooth handles, 4-5 points)
DROP_RAMP = [(0, 0.3), (0.55, 0.32), (0.8, 1.2), (1.0, 2.2)]
MID_RAMP = [(0, 2.0), (0.38, 1.4), (0.5, 0.3), (0.62, 0.32), (1.0, 2.0)]   # slow ON the beat
END_RAMP = [(0, 1.6), (0.12, 0.6), (0.3, 0.28), (1.0, 0.25)]


def S(clip, start, end, src, speed=1.0, zoom=(1.0, 1.06), center=(0.5, 0.5), fx=()):
    return dict(clip=clip, start=start, end=end, src=src, speed=speed, zoom=zoom,
                center=center, fx=set(fx))


SHOTS = [
    # build: 2 long holds, already moving
    S(2, 0, 4, 0.2, 0.9, (1.12, 1.0), (0.45, 0.55), ['fadein']),
    S(3, 4, 7.5, 1.0, 0.9, (1.0, 1.12), (0.45, 0.55), ['zoom_out_tr']),
    S(3, 7.5, 8, None, 0, (1.12, 1.14), (0.45, 0.55), ['freeze', 'dark']),
    # DROP: strongest, widest shot, flash + shake + ramp
    S(1, 8, 12, 0.6, DROP_RAMP, (1.12, 1.0), (0.45, 0.55), ['hit', 'zoom_land']),
    # 2-beat alternation, clean cuts
    S(4, 12, 14, 0.4, 1.0, (1.0, 1.12), (0.5, 0.55), ['glitch_in']),
    S(7, 14, 16, 0.4, 1.0, (1.0, 1.12), (0.5, 0.55), ['glitch_in']),
    S(5, 16, 18, 0.9, MID_RAMP, (1.0, 1.1), (0.45, 0.55), ['glitch_in']),
    S(2, 18, 20, 2.3, 1.0, (1.0, 1.12), (0.5, 0.55), ['glitch_in']),
    S(6, 20, 22, 1.0, MID_RAMP, (1.0, 1.1), (0.5, 0.5), ['glitch_in']),
    S(3, 22, 24, 2.7, 1.0, (1.0, 1.12), (0.45, 0.55), ['glitch_in', 'whip_out']),
    # burst: 1 beat each, wide readable shots only
    S(5, 24, 25, 3.2, 1.3, (1.0, 1.05), (0.45, 0.55), ['hit_small', 'whip_in']),
    S(1, 25, 26, 3.0, 1.3, (1.0, 1.05), (0.45, 0.55)),
    S(6, 26, 27, 3.2, 1.3, (1.0, 1.05), (0.5, 0.5)),
    S(4, 27, 28, 1.8, 1.3, (1.0, 1.08), (0.5, 0.55)),
    # closing: hero shot, final hit, slow-mo, fade
    S(4, 28, 32 + M.TAIL / BEAT + 0.1, 2.6, END_RAMP, (1.1, 1.0), (0.5, 0.52), ['hit', 'fadeout']),
]


def build_time_maps():
    for idx, sh in enumerate(SHOTS):
        t0, t1 = sh['start'] * BEAT, sh['end'] * BEAT
        nfr = max(1, int(round(t1 * FPS)) - int(round(t0 * FPS)))
        sh['t0'], sh['t1'] = t0, t1
        if 'freeze' in sh['fx']:
            sh['srcmap'] = np.full(nfr, SHOTS[idx - 1]['srcmap'][-1])
            sh['spmap'] = np.zeros(nfr)
            continue
        sub = 8
        dt = (t1 - t0) / (nfr * sub)
        us = (np.arange(nfr * sub) + 0.5) / (nfr * sub)
        sp = np.array([speed_at(sh['speed'], u) for u in us])
        cum = np.concatenate([[0], np.cumsum(sp * dt)])[::sub][:nfr]
        dur = (len(clips[sh['clip']]) - 3) / SRC_FPS
        total = cum[-1] + sp[-sub:].mean() * dt * sub
        src = sh['src']
        if src + total > dur:
            print(f'  shot {idx}: clip{sh["clip"]} in-point {src} -> {max(0, dur - total):.2f}', file=sys.stderr)
            src = max(0.0, dur - total)
        sh['srcmap'] = src + cum
        sh['spmap'] = sp[::sub][:nfr]


def fetch_blur(clip, t, speed):
    """Source frame at time t with a 180-degree shutter: blend the frames the
    shutter would have seen. At speed <= 1 this is just the (interpolated) frame."""
    arr = clips[clip]
    span = speed / FPS * 0.5 * SRC_FPS          # source frames covered by the shutter
    n = int(min(8, max(1, math.ceil(span))))
    f0 = t * SRC_FPS
    if n == 1:
        i = min(max(int(math.floor(f0)), 0), len(arr) - 2)
        a = f0 - i
        if a < 0.2 or a > 0.8:
            return np.asarray(arr[i + (a > 0.8)])
        return cv2.addWeighted(np.asarray(arr[i]), 1 - a, np.asarray(arr[i + 1]), a, 0)
    acc = np.zeros(arr.shape[1:], np.float32)
    for j in range(n):
        i = min(max(int(round(f0 + span * j / n)), 0), len(arr) - 1)
        acc += arr[i]
    return (acc / n).astype(np.uint8)


# ---------------------------------------------------------------- clean grade

def make_luts():
    x = np.arange(256) / 255.0
    y = 0.02 + x * 0.98                                   # lift shadows a touch
    y = y - 0.10 * np.clip((y - 0.72) / 0.28, 0, 1) ** 1.6  # pull highlights down
    y = 0.5 + (y - 0.5) * 1.28                             # strong contrast
    y = np.clip(y, 0, 1)
    y = y ** 1.04
    r = np.clip(y * 1.0, 0, 1)
    g = np.clip(y * 0.995, 0, 1)
    b = np.clip(y * 1.05 + 0.03 * (1 - y) * (y > 0.05), 0, 1)  # cold
    to = lambda c: np.clip(c * 255, 0, 255).astype(np.uint8)
    return to(r), to(g), to(b)


LR, LG, LB = make_luts()
SAT = np.clip(np.arange(256) * 1.22 - 0.0008 * np.arange(256) ** 2 * 0.3, 0, 255).astype(np.uint8)


def grade(img):
    r, g, b = cv2.split(img)
    img = cv2.merge([cv2.LUT(r, LR), cv2.LUT(g, LG), cv2.LUT(b, LB)])
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]
    red = (((h <= 8) | (h >= 170)) & (hsv[..., 1] > 80)).astype(np.float32)
    red = cv2.GaussianBlur(red, (0, 0), 1.5)
    hsv[..., 1] = np.clip(hsv[..., 1] * (0.55 + 0.6 * red), 0, 255).astype(np.uint8)  # low-sat, red kept
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # soft glow only on actual light sources
    lum = img.max(axis=2)
    if lum.max() > 235:
        hi = cv2.bitwise_and(img, img, mask=(lum > 235).astype(np.uint8))
        small = cv2.GaussianBlur(cv2.resize(hi, (img.shape[1] // 4, img.shape[0] // 4)), (0, 0), 5)
        img = cv2.add(img, cv2.resize(small, (img.shape[1], img.shape[0])) // 2)
    return img


yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIG = (1 - 0.3 * np.clip((_d - 0.6) / 0.8, 0, 1) ** 1.5)[..., None].astype(np.float32)
del _d, xx, yy
GRAIN = [np.random.default_rng(i).normal(0, 1, (H // 2, W // 2)).astype(np.float32) for i in range(8)]


def shot_at(t):
    for i, sh in enumerate(SHOTS):
        if sh['t0'] - 1e-6 <= t < sh['t1']:
            return i, sh
    return len(SHOTS) - 1, SHOTS[-1]


def shake_offsets(t_hit, t, amp):
    """S_Shake-style: amplitude decays to 0 over ~20 frames @30fps, ~15 Hz."""
    dtt = t - t_hit
    if dtt < 0 or dtt > 0.7:
        return 0.0, 0.0, 0.0
    a = amp * (1 - smooth(dtt / 0.7))
    w = 2 * math.pi * 15
    return (a * 14 * math.sin(w * dtt + 1.3) + a * 6 * math.sin(w * 1.7 * dtt),
            a * 20 * math.sin(w * 0.9 * dtt + 0.4),
            a * 0.9 * math.sin(w * 0.8 * dtt + 2.0))


def render_frame(fi):
    t = fi / FPS
    idx, sh = shot_at(t)
    k = max(0, min(int(round(t * FPS)) - int(round(sh['t0'] * FPS)), len(sh['srcmap']) - 1))
    u = (t - sh['t0']) / (sh['t1'] - sh['t0'])
    since, to_end = t - sh['t0'], sh['t1'] - t
    prev = SHOTS[idx - 1] if idx > 0 else None
    fx = sh['fx']

    src = grade(fetch_blur(sh['clip'], sh['srcmap'][k], sh['spmap'][k]))
    z0, z1 = sh['zoom']
    nb = max(1, round(sh['end'] - sh['start']))
    if 'freeze' in fx or nb < 2 or t >= 28 * BEAT:
        prog = smooth(u)
    else:   # step zoom: quick ease-out move right after every beat, hold in between
        bpos = (t - sh['t0']) / BEAT
        bi = min(int(bpos), nb - 1)
        frac = bpos - bi
        prog = (bi + 1 - (1 - min(frac / 0.35, 1)) ** 3) / nb
    zoom = z0 + (z1 - z0) * prog
    # scale bounce 100 -> 115 -> 100 on the big hits
    if ('hit' in fx or 'hit_small' in fx) and since < 0.35:
        zoom *= 1 + 0.15 * math.sin(math.pi * since / 0.35) * (1 if 'hit' in fx else 0.6)
    cx, cy = sh['center']

    # transitions (only at section changes)
    TR = 0.2
    zb = 0.0
    if 'zoom_out_tr' in fx and to_end < TR:
        p = smooth(1 - to_end / TR)
        zoom *= 1 + 0.7 * p * p
        zb = p
    if 'zoom_land' in fx and since < TR:
        p = 1 - smooth(since / TR)
        zoom *= 1 + 0.45 * p
        zb = max(zb, p)
    dy = 0.0
    whip = 0.0
    if 'whip_out' in fx and to_end < 0.13:
        whip = smooth(1 - to_end / 0.13)
        dy = -whip * H * 0.35
    if 'whip_in' in fx and since < 0.13:
        whip = 1 - smooth(since / 0.13)
        dy = whip * H * 0.35

    # shake only on the big hits
    dx = rot = 0.0
    amp = 1.0 if 'hit' in fx else 0.6 if 'hit_small' in fx else 0.0
    shaking = amp > 0 and since < 0.7
    if shaking:
        sx, sy, sr = shake_offsets(sh['t0'], t, amp)
        dx += sx
        dy += sy
        rot += sr

    if shaking and since < 0.45:
        # motion-blurred shake: average 3 sub-frame positions
        acc = np.zeros((H, W, 3), np.float32)
        for j in range(3):
            tj = t + (j - 1) / FPS / 3
            sx, sy, sr = shake_offsets(sh['t0'], tj, amp)
            acc += warp(src, zoom, cx, cy, sx, dy - shake_offsets(sh['t0'], t, amp)[1] + sy, sr)
        frame = (acc / 3).astype(np.uint8)
    else:
        frame = warp(src, zoom, cx, cy, dx, dy, rot)

    if zb > 0.05:
        acc = frame.astype(np.float32)
        for j in range(1, 6):
            acc += warp(src, zoom * (1 + 0.03 * j * zb), cx, cy, dx, dy, rot)
        frame = (acc / 6).astype(np.uint8)
    if whip > 0.05:
        ks = int(9 + whip * 120) | 1
        frame = cv2.filter2D(frame, -1, np.full((ks, 1), 1.0 / ks, np.float32))

    glitch_amt = 0.0
    if ('glitch_in' in fx or 'hit_small' in fx) and since < 0.07:
        glitch_amt = 1 - since / 0.07
    if glitch_amt > 0:
        r = np.random.default_rng(fi)
        frame = frame.copy()
        for _ in range(int(5 + 8 * glitch_amt)):
            y0 = int(r.integers(0, H - 30))
            hh = int(r.integers(14, 140))
            sx = int(r.normal(0, 90 * glitch_amt))
            frame[y0:y0 + hh] = np.roll(frame[y0:y0 + hh], sx, axis=1)
    blur = cv2.GaussianBlur(frame, (0, 0), 1.4)
    frame = cv2.addWeighted(frame, 1.4, blur, -0.4, 0)

    f = frame.astype(np.float32)
    # subtle chromatic fringe only during transitions
    px = int(round(10 * max(zb, whip) + 16 * glitch_amt))
    if px:
        f[..., 0] = np.roll(f[..., 0], px, axis=1)
        f[..., 2] = np.roll(f[..., 2], -px, axis=1)
    f *= VIG

    if 'hit' in fx and since < 0.2:            # white flash, ~3 frames @30fps
        a = math.exp(-since / 0.04)
        f = f * (1 - a) + 255 * a
    if 'freeze' in fx:
        g = f.mean(axis=2, keepdims=True)
        f = (f * 0.3 + g * 0.7) * 0.6
    if 'fadein' in fx and since < 0.25:
        f *= smooth(since / 0.25)
    if 'fadeout' in fx and TOTAL - t < 0.8:
        f *= smooth((TOTAL - t) / 0.8)

    gr = cv2.resize(GRAIN[fi % len(GRAIN)], (W, H), interpolation=cv2.INTER_LINEAR)
    f += gr[..., None] * 2.5
    return np.clip(f, 0, 255).astype(np.uint8)


def _work(fi):
    return render_frame(fi).tobytes()


def main(out_path, preview=False):
    audio = os.path.join(ROOT, 'work', 'music_v3.wav')
    M.build(audio)
    build_time_maps()
    nframes = int(TOTAL * FPS)
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-i', audio,
           '-c:v', 'libx264', '-preset', 'veryfast' if preview else 'slow',
           '-crf', '23' if preview else '17', '-maxrate', '25M', '-bufsize', '50M',
           '-pix_fmt', 'yuv420p', '-profile:v', 'high',
           '-c:a', 'aac', '-b:a', '256k', '-shortest', '-movflags', '+faststart', out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for buf in pool.imap(_work, range(nframes), chunksize=4):
            p.stdin.write(buf)
    p.stdin.close()
    p.wait()
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'out', 'edit_v4.mp4'),
         preview='--preview' in sys.argv)
