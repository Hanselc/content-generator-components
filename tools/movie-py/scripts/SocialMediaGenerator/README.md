# SocialMediaGenerator

Builds a social-media-style video: an opening composition (3 balanced Voronoi
shards with Ken Burns animation + title overlay) followed by a captioned
slideshow of the remaining images, with silence-padded audio and crossfades.
The video resolution is taken from the first image; audio is silence-padded
around each clip and slides crossfade entirely within the silent tails.

Slide display and crossfade durations are configurable via `params`; the intro
falls back to `INTRO_SECONDS` (from `make_video`) when the title has no audio.

---

## Input

### `movie.json` (the spec, passed via `ctx.spec_path`)

```json
{
  "title": {
    "text": "Verano en la Costa",
    "audio": "intro.mp3"
  },
  "images": [
    { "image": "01_beach.png", "text": "Llegada a la playa", "audio": "01.mp3" },
    { "image": "02_boat.png",  "text": "Paseo en barco",     "audio": "02.mp3" },
    { "image": "03_food.png",  "text": "Comida local",       "audio": "03.mp3" },
    { "image": "04_sunset.png" }
  ]
}
```

`title` may also be a plain string (no intro audio):

```json
{ "title": "Verano en la Costa", "images": [ ... ] }
```

### `ctx.input_folder`

Must contain every image and audio file referenced by the spec:

```text
input_folder/
├── 01_beach.png
├── 02_boat.png
├── 03_food.png
├── 04_sunset.png
├── intro.mp3
├── 01.mp3
├── 02.mp3
└── 03.mp3
```

### `ctx.params`

Validated against `PARAM_SCHEMA` (`type: object`,
`additionalProperties: false`); both keys are optional with defaults:

| Param | Type | Default | Constraint | Meaning |
|---|---|---|---|---|
| `display_seconds` | number | `make_video.DEFAULT_DISPLAY` (5) | `minimum: 0.1` | Seconds each slide displays when it has no audio |
| `transition_seconds` | number | `make_video.DEFAULT_TRANSITION` (1) | `minimum: 0` | Crossfade duration in seconds |

Empty object `{}` is valid and uses both defaults.

### Field reference

**Required**

- `title` (string **or** object) — the on-screen title.
  - As object: `text` (string, non-empty, required) + optional `audio`
    (filename resolved against `ctx.input_folder`).
  - As string: used as `text` with no intro audio.
- `images` (array, **at least 2 entries**) — each entry:
  - `image` (filename, required) — must exist in `ctx.input_folder`.
  - The first image also defines the output video resolution.

**Optional**

- `images[].text` (string) — caption overlaid on the slide; omitted = no
  caption.
- `images[].audio` (filename) — narration/sound for the slide, resolved
  against `ctx.input_folder`; omitted/empty = the slide plays silently for
  `display_seconds + 2 * transition_seconds`.

> A slide with audio has duration `audio.duration + 2 * transition_seconds`
> (silent padding before/after). A slide without audio has duration
> `display_seconds + 2 * transition_seconds`.

---

## Output

Suppose the title audio is 8.0s, slides 01–03 have audio of 6.0s / 5.0s /
7.0s, slide 04 has no audio, and `transition_seconds` = 1.0,
`display_seconds` = 5.0. Then:
- intro section ≈ 8.0 + 1.0 = 9.0s (title audio + one transition pad),
- slides start at `intro_duration - transition_seconds` = 8.0s on the
  timeline and each step by `prev_duration - transition_seconds`.

### Files written to `ctx.output_folder`

```text
output_folder/
└── 20260701_120000.mp4
```

The filename is a UTC timestamp `YYYYMMDD_HHMMSS.mp4` (assigned by
`make_video.assemble`). The MP4 is h264 / yuv420p with AAC audio, rendered at
`make_video.FPS` fps and at the resolution of the first image (e.g.
`1920x1080`).

### `build()` return value (the result dict)

```json
{
  "script_id": "SocialMediaGenerator",
  "spec_path": "/abs/path/to/movie.json",
  "started_at": "2026-07-01T12:00:00Z",
  "finished_at": "2026-07-01T12:01:30Z",
  "total_duration_seconds": 35.0,
  "sections": [
    { "name": "intro",  "start_seconds": 0.0,  "end_seconds": 9.0,
      "duration_seconds": 9.0,  "started_at": "...", "finished_at": "..." },
    { "name": "slides", "start_seconds": 8.0,  "end_seconds": 35.0,
      "duration_seconds": 27.0, "started_at": "...", "finished_at": "..." }
  ],
  "outputs": [
    { "index": 0, "path": "/abs/output_folder/20260701_120000.mp4",
      "kind": "video", "label": "main" }
  ],
  "metadata": {
    "frame_count": 1050,
    "fps": 30,
    "resolution": "1920x1080",
    "codec": "h264",
    "file_size_bytes": 52428800,
    "image_count": 4,
    "intro_seconds": 9.0,
    "title": "Verano en la Costa",
    "title_audio": "intro.mp3",
    "transition_seconds": 1.0,
    "display_seconds": 5.0
  }
}
```

> `total_duration_seconds`, `frame_count`, `file_size_bytes`, and the
> `started_at` / `finished_at` timestamps above are illustrative — actual
> values depend on the probed audio durations, the first image's resolution,
> and render wall-clock time. `metadata.title_audio` is `null` when the title
> has no audio.

---

## Key points

- **Input** — a `movie.json`-style spec with required `title` (string or
  `{text, audio}`) and `images` (≥2 entries, each with required `image` and
  optional `text`/`audio`). Params tune `display_seconds` and
  `transition_seconds`; everything else is internal.
- **Output** — a single timestamped `.mp4` (h264/yuv420p, AAC, first image's
  resolution) returned via one `outputs` entry of kind `video`, label `main`.
- **Timeline** — two sections: `intro` (Voronoi/Ken Burns composition + title
  overlay, duration = title audio + one transition pad, or `INTRO_SECONDS`
  when the title is silent) and `slides` (captioned slideshow with crossfades
  starting at `intro_duration - transition_seconds`).