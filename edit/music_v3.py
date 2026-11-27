"""Version 3 soundtrack, arranged the way edit tutorials structure an edit (135 BPM):

  0-8    build: filtered groove + melody, riser, half-beat silence
  8      DROP (impact + crash)
  8-24   full groove (2-beat cut section)
  24-28  burst bar with a clap roll
  28     final hit, then melody + long 808 ringing out
"""
import wave

import numpy as np

from music import clap, fft_filter, hat, impact, note_hz, place, reverb, riser, whoosh
from music_v2 import BASS, MEL, bass_dist, bell, crash, kick_hard

BPM = 135
BEAT = 60 / BPM
STEP = BEAT / 4
BAR = BEAT * 4
SR = 44100
DROP = 8
BURST = 24
END_HIT = 28
END = 32
TAIL = 1.0
TOTAL = END * BEAT + TAIL

KICK_STEPS = [0, 3, 6, 10]
CLAP_STEPS = [4, 12]


def kick_times(b0=DROP, b1=END_HIT):
    return [bar * BAR + s * STEP for bar in range(b0 // 4, b1 // 4) for s in KICK_STEPS]


def build(path):
    N = int(TOTAL * SR) + SR
    mel, drums, bass, fx, build_drums = (np.zeros(N) for _ in range(5))

    for ph in range(END // 8):
        for s, nn in enumerate(MEL):
            if nn:
                place(mel, bell(note_hz(nn)), ph * 2 * BAR + s * STEP, 1.0 if s % 4 == 0 else 0.8)

    # build: same groove, but it will be low-passed (classic filter build)
    for t in kick_times(0, DROP):
        place(build_drums, kick_hard(), t, 0.9)
    for bar in range(0, DROP // 4):
        for s in CLAP_STEPS:
            place(build_drums, clap(), bar * BAR + s * STEP, 0.7)

    for t in kick_times():
        place(drums, kick_hard(), t)
    for bar in range(DROP // 4, END_HIT // 4):
        for s in CLAP_STEPS:
            place(drums, clap(), bar * BAR + s * STEP, 0.85)
        for s in range(16):
            place(drums, hat(), bar * BAR + s * STEP, 0.45 if s % 2 == 0 else 0.22)
    # clap roll leading out of the burst bar into the final hit
    for k in range(8):
        place(drums, clap(), (END_HIT - 1) * BEAT + k * STEP / 2, 0.35 + 0.06 * k)

    for ph in range(DROP // 8, END_HIT // 8 + 1):
        for step, nn, ln in BASS:
            t = ph * 2 * BAR + step * STEP
            if DROP * BEAT <= t < END_HIT * BEAT:
                place(bass, bass_dist(note_hz(nn), ln * STEP), t)
    place(bass, bass_dist(note_hz('F1'), 3.2), END_HIT * BEAT, 0.9)    # long ringing 808

    duck = np.ones(N)
    for t in kick_times():
        i = int(t * SR)
        j = min(N, i + int(0.2 * SR))
        duck[i:j] = np.minimum(duck[i:j], 1 - 0.55 * np.exp(-np.arange(j - i) / SR / 0.06))

    drop_i, end_i = int(DROP * BEAT * SR), int(END_HIT * BEAT * SR)
    gap_i = int((DROP - 0.5) * BEAT * SR)
    # build section filtered, opening up slightly over the 2 bars
    lo = fft_filter(mel + build_drums * 1.2, hi=700)
    mid = fft_filter(mel + build_drums * 1.2, hi=2200)
    ramp = np.clip(np.arange(N) / max(gap_i, 1), 0, 1)
    build_mix = lo * (1 - ramp) + mid * ramp
    melmix = mel * duck
    melmix[:drop_i] = build_mix[:drop_i]
    melmix[gap_i:drop_i] = 0
    outro = fft_filter(mel, hi=1600)
    melmix[end_i:] = outro[end_i:]

    place(fx, riser(BAR * 1.2), (DROP - 0.5) * BEAT - BAR * 1.2, 0.6)
    place(fx, impact(), DROP * BEAT, 1.0)
    place(fx, crash(), DROP * BEAT, 1.0)
    place(fx, whoosh(0.3), BURST * BEAT - 0.3, 0.6)
    place(fx, impact(), END_HIT * BEAT, 0.9)
    place(fx, crash(), END_HIT * BEAT, 0.8)

    mix = reverb(melmix, 1.5, 0.25) * 0.5 + drums * 0.8 + bass * duck * 0.75 + fx * 0.7
    mix /= np.percentile(np.abs(mix), 99.9) + 1e-9
    mix = np.tanh(mix * 1.4) / np.tanh(1.4)
    total = int(TOTAL * SR)
    mix = mix[:total]
    f = int(0.8 * SR)
    mix[-f:] *= np.linspace(1, 0, f) ** 1.5
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
    build(sys.argv[1] if len(sys.argv) > 1 else 'music_v3.wav')
