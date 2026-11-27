"""Original phonk-style beat synthesized from scratch (no samples, no copyrighted audio).

Cowbell melody + distorted 808 + kick/clap/hats, with a muffled intro, a riser,
a half-beat of silence before the drop, sidechain pumping and a reverb tail.
"""
import numpy as np

from config import (BAR, BEAT, BREAK_START, INTRO_END, OUTRO, SONG_END, SR, STEP,
                    TOTAL, kick_times, snare_times)

rng = np.random.default_rng(7)
N = int(TOTAL * SR) + SR


def note_hz(name):
    names = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7,
             'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    n, octv = name[:-1], int(name[-1])
    midi = 12 * (octv + 1) + names[n]
    return 440.0 * 2 ** ((midi - 69) / 12)


def place(buf, sig, t, gain=1.0):
    i = int(round(t * SR))
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


def env(n, attack=0.002, decay=0.2):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / decay)


def fft_filter(x, lo=None, hi=None, soft=1.15):
    """Static band filter in the frequency domain with soft (butterworth-like) edges."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    if hi is not None:
        g *= 1 / np.sqrt(1 + (f / hi) ** (8 * soft))
    if lo is not None:
        g *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (8 * soft))
    return np.fft.irfft(X * g, n=len(x))


def reverb(x, seconds=1.8, mix=0.25, pre=0.012):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t * 6.9 / seconds)
    ir = fft_filter(ir, lo=200, hi=7000)
    ir[: int(pre * SR)] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    L = len(x) + n
    nfft = 1 << (L - 1).bit_length()
    wet = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[: len(x)]
    return x * (1 - mix) + wet * mix * 3.0


# ---------------------------------------------------------------- instruments

def cowbell(freq, dur=0.32):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sq = lambda f: np.sign(np.sin(2 * np.pi * f * t))
    s = 0.6 * sq(freq) + 0.4 * sq(freq * 1.4836)
    s = fft_filter(s, lo=freq * 0.7, hi=freq * 4.5)
    e = env(n, 0.001, 0.055) * 0.6 + env(n, 0.001, 0.2) * 0.4
    return np.tanh(2.2 * s * e) * 0.8


def kick(dur=0.45):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 48 + 170 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * env(n, 0.001, 0.16)
    click = rng.standard_normal(n) * env(n, 0.0005, 0.004) * 0.4
    return np.tanh(2.5 * (s + click))


def bass808(freq, dur, glide_from=None):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = np.full(n, freq)
    if glide_from:
        g = np.exp(-t / 0.06)
        f = freq + (glide_from - freq) * g
    else:
        f = freq * (1 + 0.9 * np.exp(-t / 0.02))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + 0.25 * np.sin(2 * ph)
    e = np.clip(t / 0.003, 0, 1) * np.exp(-t / (dur * 0.9))
    rel = np.clip((dur - t) / 0.02, 0, 1)
    return np.tanh(3.2 * s * e * rel) * 0.9


def clap(dur=0.35):
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    noise = fft_filter(noise, lo=900, hi=9000)
    e = np.zeros(n)
    for k, off in enumerate([0, 0.011, 0.022]):
        i = int(off * SR)
        e[i:] += env(n - i, 0.0005, 0.008 if k < 2 else 0.11)
    return np.tanh(noise * e * 1.6) * 0.8


def hat(open_=False):
    dur = 0.25 if open_ else 0.06
    n = int(dur * SR)
    noise = fft_filter(rng.standard_normal(n), lo=7000)
    return noise * env(n, 0.0005, 0.07 if open_ else 0.014) * 0.6


def riser(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    noise = rng.standard_normal(n)
    # crossfade of progressively brighter bands = upward sweep
    bands = [fft_filter(noise, lo=200 * 2 ** k, hi=600 * 2 ** k) for k in range(6)]
    out = np.zeros(n)
    pos = u * (len(bands) - 1)
    for k, b in enumerate(bands):
        w = np.clip(1 - np.abs(pos - k), 0, 1)
        out += b * w
    f = 200 + 1800 * u ** 2
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.15
    return (out * 0.5 + tone) * u ** 2


def impact():
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    f = 30 + 90 * np.exp(-t / 0.08)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.7)
    noise = fft_filter(rng.standard_normal(n), lo=150, hi=5000) * env(n, 0.001, 0.25)
    return np.tanh(1.8 * (boom + 0.6 * noise)) * 0.9


def whoosh(dur=0.35):
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    s = fft_filter(rng.standard_normal(n), lo=500, hi=6000)
    return s * np.sin(np.pi * u) ** 2 * 0.35


# ------------------------------------------------------------------ patterns

MEL_A = ['C#5', None, 'C#5', None, 'E5', None, 'C#5', None,
         'G#4', None, 'G#4', 'B4', None, 'C#5', None, None]
MEL_B = ['C#5', None, 'C#5', None, 'E5', None, 'F#5', None,
         'E5', None, 'C#5', None, 'B4', None, 'G#4', None]

# 808 line per 2-bar phrase: (step, note, length in steps, glide_from)
BASS = [(0, 'C#2', 6, None), (6, 'C#2', 4, None), (10, 'E2', 6, 'C#2'),
        (16, 'C#2', 6, None), (22, 'C#2', 4, None), (26, 'A1', 6, 'C#2')]


def build(path):
    mel = np.zeros(N)
    drums = np.zeros(N)
    bass = np.zeros(N)
    fx = np.zeros(N)

    # cowbell melody across the whole song (with variation every other bar)
    for bar in range(int(SONG_END / 4)):
        pat = MEL_A if bar % 2 == 0 else MEL_B
        for s, nn in enumerate(pat):
            t = bar * BAR + s * STEP
            if nn is None or t >= BREAK_START * BEAT and t < INTRO_END * BEAT:
                continue
            vel = 1.0 if s % 4 == 0 else 0.8
            place(mel, cowbell(note_hz(nn)), t, vel)
            if bar >= INTRO_END / 4:   # octave layer on the drop
                place(mel, cowbell(note_hz(nn) * 2), t, 0.25 * vel)

    # drums / 808 only between the drop and the outro
    for t in kick_times():
        place(drums, kick(), t, 1.0)
    for t in snare_times():
        place(drums, clap(), t, 0.8)
    for bar in range(int(INTRO_END / 4), int(OUTRO / 4)):
        for s in range(0, 16, 2):
            place(drums, hat(open_=(s == 14 and bar % 2 == 1)), bar * BAR + s * STEP, 0.55)
        if bar % 2 == 1:   # triplet hat roll at the end of each phrase
            for k in range(6):
                place(drums, hat(), bar * BAR + 12 * STEP + k * (BEAT / 6), 0.35 + 0.05 * k)
    for ph in range(int(INTRO_END / 8), int(OUTRO / 8)):
        for step, nn, ln, gl in BASS:
            t = ph * 2 * BAR + step * STEP
            place(bass, bass808(note_hz(nn), ln * STEP, note_hz(gl) if gl else None), t, 1.0)

    # sidechain pump: duck melody + bass after every kick
    duck = np.ones(N)
    for t in kick_times():
        i = int(t * SR)
        n = int(0.22 * SR)
        j = min(N, i + n)
        curve = 1 - 0.65 * np.exp(-np.arange(j - i) / SR / 0.07)
        duck[i:j] = np.minimum(duck[i:j], curve)

    # intro is muffled (lowpass), opens up for the drop
    drop_i = int(INTRO_END * BEAT * SR)
    muffled = fft_filter(mel, hi=900)
    blend = np.zeros(N)
    blend[drop_i:] = 1.0
    out_i = int(OUTRO * BEAT * SR)
    blend[out_i + int(0.05 * SR):] = 0.0   # outro muffled again
    mel = muffled * (1 - blend) + mel * blend * duck

    # FX: riser into the drop, impacts on drop + outro, whooshes on velocity cuts
    rs = 2 * BAR - 0.5 * BEAT
    place(fx, riser(rs), INTRO_END * BEAT - 0.5 * BEAT - rs, 0.55)
    place(fx, impact(), INTRO_END * BEAT, 0.9)
    place(fx, impact(), OUTRO * BEAT, 0.8)
    for b in (25.6, 27.6, 29.6, 31.6):
        place(fx, whoosh(0.4), b * BEAT, 0.8)

    mix = reverb(mel, 1.6, 0.22) * 0.55 + drums * 0.75 + bass * duck * 0.8 + reverb(fx, 2.2, 0.3) * 0.7

    # master: glue saturation, fade, normalise
    mix /= np.percentile(np.abs(mix), 99.9) + 1e-9
    mix = np.tanh(mix * 1.4) / np.tanh(1.4)
    end = int(TOTAL * SR)
    fade = int(0.6 * SR)
    mix[end - fade:end] *= np.linspace(1, 0, fade)
    mix[end:] = 0
    mix = mix[:end]
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 0.93

    # cheap stereo width: small delay on one side for the melody-ish highs
    side = fft_filter(mix, lo=2500) * 0.15
    d = int(0.011 * SR)
    left = mix + np.concatenate([np.zeros(d), side[:-d]])
    right = mix - np.concatenate([np.zeros(d), side[:-d]])
    st = np.stack([left, right], 1)
    st /= np.max(np.abs(st)) / 0.95

    import wave
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype('<i2').tobytes())
    return path


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else 'music.wav')
