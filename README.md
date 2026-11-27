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

## Versione 2 – [`out/edit_v2.mp4`](out/edit_v2.mp4)
Più dura e più cinematografica, ~15s, nessuna scritta:
- parte subito sul colpo (niente intro lenta) e finisce sulla stessa inquadratura dell'inizio, così il video va in loop
- base funk/phonk a 135 BPM con 808 distorto, campane FM e clap (`edit/music_v2.py`)
- tagli su cassa e clap, nel secondo drop anche a mezzo battito
- grading freddo con contrasto alto e bagliore sulle luci del distributore
- pausa a metà: slow-mo con bande cinema, tutto in bianco e nero tranne il rosso, poi silenzio e secondo drop
- transizioni: zoom, whip verticale, stacco al nero, glitch, effetto scattoso a fotogrammi

```bash
python3 edit/render_v2.py out/edit_v2.mp4
```
