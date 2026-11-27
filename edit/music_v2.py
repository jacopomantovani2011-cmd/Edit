"""Version 2 soundtrack: hard 'funk/phonk montagem' groove, 135 BPM, starts on the hit.

Original synthesis only. Structure (beats):
  0-16   drop 1 (starts immediately, no intro)
  16-20  break: FM bell melody alone, filtered, riser
  20-32  drop 2 (harder, extra claps, crash)
  32     final hit, hard stop (loops back to the start)
"""
import wave

import numpy as np

from music import clap, env, fft_filter, hat, impact, note_hz, place, reverb, riser, rng, whoosh

BPM = 135
BEAT = 60 / BPM
STEP = BEAT / 4
BAR = BEAT * 4
SR = 44100
DROP1 = (0, 16)
BREAK = (16, 20)
DROP2 = (20, 32)
END = 32
TAIL = 0.45
TOTAL = END * BEAT + TAIL

KICK_STEPS = [0, 3, 6, 10]          # tamborzao-like
CLAP_STEPS = [4, 12]
CLAP_EXTRA = [14]                   # drop 2 only


def drum_bars():
    for b0, b1 in (DROP1, DROP2):
        for bar in range(b0 // 4, b1 // 4):
            yield bar, b0 == DROP2[0]


def kick_times():
    return [bar * BAR + s * STEP for bar, _ in drum_bars() for s in KICK_STEPS]


def clap_times():
    out = []
    for bar, hard in drum_bars():
        steps = CLAP_STEPS + (CLAP_EXTRA if hard else [])
        out += [bar * BAR + s * STEP for s in steps]
    return out


def kick_hard(dur=0.38):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 52 + 240 * np.exp(-t / 0.028)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.0005, 0.12)
    click = fft_filter(rng.standard_normal(n), lo=2000) * env(n, 0.0003, 0.003)
    return np.clip(np.tanh(4.0 * (s + 0.5 * click)) * 1.1, -1, 1)


def bass_dist(freq, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = freq * (1 + 1.2 * np.exp(-t / 0.015))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph)
    e = np.clip(t / 0.002, 0, 1) * np.exp(-t / (dur * 0.7)) * np.clip((dur - t) / 0.015, 0, 1)
    d = np.tanh(8 * s * e)                       # hard distortion -> buzzy funk bass
    return fft_filter(d, hi=3500) * 0.85


def bell(freq, dur=0.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    idx = 3.0 * np.exp(-t / 0.12)
    s = np.sin(2 * np.pi * freq * t + idx * np.sin(2 * np.pi * freq * 3.5 * t))
    s += 0.3 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-t / 0.08)
    return s * env(n, 0.001, 0.28) * 0.6


def crash():
    n = int(2.0 * SR)
    s = fft_filter(rng.standard_normal(n), lo=4000)
    return s * env(n, 0.001, 0.6) * 0.35


# F minor, 2-bar riff in 16ths
MEL = ['F5', None, 'G#5', None, 'C6', None, 'G#5', 'G5', None, 'F5', None, 'D#5', None, 'F5', None, None,
       'F5', None, 'G#5', None, 'C6', None, 'D#6', None, 'C#6', None, 'C6', None, 'G#5', None, 'G5', None]
BASS = [(0, 'F1', 3), (3, 'F1', 3), (6, 'F1', 4), (10, 'G#1', 6),
        (16, 'F1', 3), (19, 'F1', 3), (22, 'C#2', 4), (26, 'C2', 6)]


def build(path):
    N = int(TOTAL * SR) + SR
    mel, drums, bass, fx = (np.zeros(N) for _ in range(4))

    for ph in range(END // 8):
        for s, nn in enumerate(MEL):
            if nn:
                place(mel, bell(note_hz(nn)), ph * 2 * BAR + s * STEP, 1.0 if s % 4 == 0 else 0.8)

    for t in kick_times():
        place(drums, kick_hard(), t, 1.0)
    for t in clap_times():
        place(drums, clap(), t, 0.85)
    for bar, hard in drum_bars():
        for s in range(16):
            place(drums, hat(), bar * BAR + s * STEP, (0.45 if s % 2 == 0 else 0.25) * (1.1 if hard else 1))
    for b0, b1 in (DROP1, DROP2):
        for ph in range(b0 // 8, (b1 + 7) // 8):
            for step, nn, ln in BASS:
                t = ph * 2 * BAR + step * STEP
                if b0 * BEAT <= t < b1 * BEAT:
                    place(bass, bass_dist(note_hz(nn), ln * STEP), t)

    duck = np.ones(N)
    for t in kick_times():
        i = int(t * SR)
        j = min(N, i + int(0.2 * SR))
        duck[i:j] = np.minimum(duck[i:j], 1 - 0.6 * np.exp(-np.arange(j - i) / SR / 0.06))

    # break: melody filtered + reverb, others silent
    b0, b1 = int(BREAK[0] * BEAT * SR), int(BREAK[1] * BEAT * SR)
    filt = fft_filter(mel, hi=1400)
    melmix = mel * duck
    melmix[b0:b1] = filt[b0:b1] * 1.3
    # half-beat of silence right before drop 2
    gap0 = int((BREAK[1] - 0.5) * BEAT * SR)
    melmix[gap0:b1] = 0

    place(fx, impact(), 0.0, 0.9)
    place(fx, crash(), 0.0, 1.0)
    place(fx, riser(BAR - 0.5 * BEAT), BREAK[0] * BEAT + 0.5 * BEAT, 0.6)
    place(fx, impact(), DROP2[0] * BEAT, 1.0)
    place(fx, crash(), DROP2[0] * BEAT, 1.0)
    place(fx, impact(), END * BEAT, 0.9)
    for b in (8, 12, 24, 28):
        place(fx, whoosh(0.3), b * BEAT - 0.3, 0.7)

    mix = reverb(melmix, 1.4, 0.25) * 0.5 + drums * 0.8 + bass * duck * 0.75 + fx * 0.7
    # hard stop on the final hit (only the impact rings briefly)
    endi = int(END * BEAT * SR)
    mix[endi:] = fx[endi:] * 0.7 + drums[endi:endi + 1].sum() * 0
    mix /= np.percentile(np.abs(mix), 99.9) + 1e-9
    mix = np.tanh(mix * 1.5) / np.tanh(1.5)
    total = int(TOTAL * SR)
    mix = mix[:total]
    f = int(0.25 * SR)
    mix[-f:] *= np.linspace(1, 0, f)
    side = fft_filter(mix, lo=2500) * 0.15
    d = int(0.011 * SR)
    sh = np.concatenate([np.zeros(d), side[:-d]])
    st = np.stack([mix + sh, mix - sh], 1)
    st /= np.max(np.abs(st)) / 0.8
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype('<i2').tobytes())


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'music_v2.wav')
