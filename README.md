# Edit TikTok – Supermoto Night

Edit verticale 9:16 (1080×1920, 30fps, ~18s) stile TikTok, montato a tempo su una base phonk originale (130 BPM) generata da zero, senza musica coperta da copyright.

**Risultato:** [`out/edit.mp4`](out/edit.mp4)

## Struttura
| Tempo | Sezione | Cosa succede |
|---|---|---|
| 0–3.5s | Intro | slow-mo, beat ovattato, riser, titolo "NIGHT RIDE" |
| 3.5–3.7s | Pausa | freeze desaturato + silenzio prima del drop |
| 3.7–11s | Drop | tagli su ogni battuta, flash, shake, zoom punch, RGB split, speed ramp |
| 11–14.8s | Velocity | rampe veloce→lento→veloce con whip pan e whoosh |
| 14.8–17.8s | Outro | slow-mo sulla moto, titolo "2 TEMPI", fade |

## Come rifarlo con altre clip
```bash
pip install numpy opencv-python-headless pillow
edit/prepare.sh clip1.mov clip2.mov ... clip7.mov   # 7 clip verticali
python3 edit/render.py out/edit.mp4                 # --preview per una prova veloce
```
- `edit/config.py` – BPM, struttura, pattern della batteria (video e audio usano gli stessi tempi)
- `edit/music.py` – sintesi della base (cowbell, 808, kick, clap, hi-hat, riser, impatti)
- `edit/render.py` – lista degli shot (`SHOTS`), titoli (`TITLES`), effetti e color grading

Font: Anton (SIL Open Font License).
