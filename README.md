# SKIP COUNT · 跳绳计数器

A camera-based jump-rope counter that runs entirely in the browser — no app, no
server. It uses on-device human **pose estimation** (MediaPipe Pose) to count jumps
from body/foot motion, with a 1-minute timed mode, live pace + vertical-height
analytics, and installable PWA support.

**Live:** https://jump-rope-site.pages.dev

<p align="center"><img src="icon-512.png" width="140" alt="icon"></p>

## Features

- **Real pose-based counting** — MediaPipe Pose Landmarker (lite) tracks 33 body
  keypoints; jumps are counted from vertical oscillation, not fragile motion heuristics.
- **Two switchable algorithms**
  - **B · 脚踝 (default)** — ankle-centre vertical signal, lookahead local-max +
    adaptive prominence, plus a **double-bounce guard** (handles "foot bounces twice,
    rope passes once"). Best all-round accuracy.
  - **A · 身体·抗绊绳** — confidence-weighted consensus of 5 body parts + a rhythm-
    stability gate that pauses/retracts on trips. Good for clean single-bounce with
    strict trip exclusion.
- **1-minute timed mode** with a 3·2·1 countdown; the clock starts on the **first
  detected jump** (get-ready time doesn't eat your minute).
- **Posture / visibility hints** — warns when you're not upright or feet aren't in frame.
- **Front/back camera switch**, sensitivity slider, sound + haptics.
- **Skeleton overlay** drawn on the live video.
- **Data export** — per-jump timing / vertical amplitude / interruptions as JSON, and a
  **pace + vertical-height chart** (per-5-second segments) rendered on-canvas to PNG.
- **PWA** — Add to Home Screen for a fullscreen app; service worker caches the shell +
  model for offline use. Web Worker keeps inference off the UI thread.

## Tech

- Single static `index.html` (inline CSS/JS). `pose-worker.js` runs MediaPipe in a
  Web Worker. `sw.js` + `manifest.webmanifest` provide offline/installable PWA.
- MediaPipe Tasks Vision (`@mediapipe/tasks-vision`) loaded from CDN; pose model from
  Google's model store, cached by the service worker after first load.
- No build step. Deploy the folder to any static host (Cloudflare Pages, Netlify,
  GitHub Pages, …).

## Deploy

```bash
# Cloudflare Pages
npx wrangler pages deploy .
```
Any static host works — the app is just files. HTTPS is required for the camera.

## Algorithm notes

The counting algorithm was developed and validated offline against real jump-rope
videos (see `tools/`). Key findings:

- **Jump rate = body vertical oscillation frequency.** Multiple independent signals
  (body centre, ankle, wrist, rope motion) agree on the fundamental; naive peak-counting
  of the rope/wrist signal over-counts ~1.4× because those carry two motion bursts per
  jump.
- **Multi-part consensus** cancels per-landmark jitter (~√N quieter) → far fewer false
  counts than a single landmark when standing still.
- **Ankle signal** has the highest SNR for each jump and, with an adaptive prominence
  threshold, cleanly separates the real jump from an intermediate "double bounce".
- **Double-bounce guard:** skip a peak whose interval is < 0.5× the recent median
  interval — validated to keep single-bounce counts intact while halving double-bounce.
- Trip/rhythm gates were tested but hurt accuracy when combined with the ankle detector
  (they misread cadence/double-bounce variation as trips), so trips are **reported**
  (in the export/chart) rather than subtracted from the count.

See `tools/README.md` for the validation scripts.

## License

MIT
