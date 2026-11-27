"""Shared timing config: music and video both read from here so cuts land on the beat."""

BPM = 130
BEAT = 60.0 / BPM          # seconds per beat
STEP = BEAT / 4            # 16th note
BAR = BEAT * 4

FPS = 30
W, H = 1080, 1920          # TikTok 9:16
SRC_FPS = 60               # sources are motion-interpolated to 60fps
SRC_W, SRC_H = 576, 1024

SR = 44100

# Song structure (in beats)
INTRO_END = 8              # drop lands here
BREAK_START = 7.5          # half-beat of silence before the drop
PART_B = 24                # velocity section
OUTRO = 32                 # final hero shot
SONG_END = 36
TAIL = 1.2                 # seconds of reverb tail / fade after the last beat

TOTAL = SONG_END * BEAT + TAIL

# Drum pattern inside one bar (16th steps)
KICK_STEPS = [0, 6, 10]
SNARE_STEPS = [4, 12]


def kick_times():
    """Times (s) of every kick in the sections where drums play."""
    out = []
    for bar in range(int(INTRO_END / 4), int(OUTRO / 4)):
        for s in KICK_STEPS:
            t = bar * BAR + s * STEP
            out.append(t)
    return out


def snare_times():
    out = []
    for bar in range(int(INTRO_END / 4), int(OUTRO / 4)):
        for s in SNARE_STEPS:
            out.append(bar * BAR + s * STEP)
    return out
