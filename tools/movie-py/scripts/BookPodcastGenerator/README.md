# BookPodcastGenerator

Builds a multi-episode book podcast from a book spec describing book-level
`welcome` / `summary_intro` / `farewell` audio plus per-chapter `content`
(and optional `summary`) audio supplied as **arrays** of files. A chapter's
audio is the concatenation of its content segments; this lets a chapter be
produced as multiple TTS segments that are joined into one continuous
chapter audio.

Episode grouping and silence timings (min/target chapter duration,
inter-segment silence, trailing silence) are script-internal constants and
are **not** configurable via the spec or params.

---

## Input

### `podcast.json` (the spec, passed via `ctx.spec_path`)

```json
{
  "book_title": "Don Quijote",
  "author": "Cervantes",
  "intro_message": "Bienvenidos al podcast de Don Quijote",
  "welcome": "welcome.wav",
  "summary_intro": "summary_intro.wav",
  "farewell": "farewell.wav",
  "chapters": [
    {
      "title": "Capítulo 1",
      "content": ["ch01a.wav", "ch01b.wav"],
      "summary": ["sum01.wav"]
    },
    {
      "title": "Capítulo 2",
      "content": ["ch02a.wav", "ch02b.wav", "ch02c.wav"],
      "summary": ["sum02a.wav", "sum02b.wav"]
    },
    {
      "title": "Capítulo 3",
      "content": ["ch03.wav"],
      "summary": []
    }
  ]
}
```

### `ctx.input_folder`

Must contain all referenced audio files:

```text
input_folder/
├── welcome.wav
├── summary_intro.wav
├── farewell.wav
├── ch01a.wav
├── ch01b.wav
├── ch02a.wav
├── ch02b.wav
├── ch02c.wav
├── ch03.wav
├── sum01.wav
├── sum02a.wav
└── sum02b.wav
```

### `ctx.params`

Empty object `{}` (the schema rejects any keys). Episode grouping and
silences use the internal constants:

| Constant | Value |
|---|---|
| `min_chapter_duration_seconds` | 300 |
| `target_chapter_duration_seconds` | 600 |
| `silence_between_segments_ms` | 1000 |
| `trailing_silence_ms` | 5000 |

### Field reference

**Required**

- `book_title` (string) — shown in the video title phase + metadata.
- `author` (string) — shown under the title in the video + metadata.
- `chapters` (non-empty array) — each chapter object:
  - `title` (string; defaults to `Chapter {i+1}` if omitted)
  - `content` (**array of filenames**, required, non-empty) — concatenated
    in order to form the chapter's content audio.
  - `summary` (**array of filenames**, optional; omitted/empty = no summary
    for that chapter) — concatenated in order.

**Optional**

- `intro_message` (string) — book-level on-screen text shown during the
  video intro phase; omitted = skip the intro phase.
- `welcome`, `summary_intro`, `farewell` (single filename each) — book-level
  audio played at the start / before summaries / at the end of every
  episode. Omit any to drop that segment.

> A chapter's content duration is the **sum** of its `content` segments'
> durations.

---

## Output

Suppose the chapter content durations are: Capítulo 1 = 320s, Capítulo 2 =
350s, Capítulo 3 = 80s. Grouping by the 600s target: episode 1 = Capítulo 1
alone, episode 2 = Capítulo 2 alone, episode 3 = Capítulo 3 (80s,
undersized → merged backward into episode 2). **Result: 2 episodes.**

### Files written to `ctx.output_folder`

```text
output_folder/
├── podcast_01.wav
├── podcast_01.mp4
├── podcast_02.wav
├── podcast_02.mp4
└── metadata.json
```

Each `podcast_NN.wav` is 16-bit PCM stereo; each `podcast_NN.mp4` is
1920x1080, 30fps, h264/yuv420p with AAC audio and the 3-phase video
(0–5s intro text → 5–10s title+author → black for the remainder; audio
plays throughout).

### `metadata.json`

```json
{
  "book_title": "Don Quijote",
  "author": "Cervantes",
  "intro_message": "Bienvenidos al podcast de Don Quijote",
  "started_at": "2026-07-01T12:00:00Z",
  "finished_at": "2026-07-01T12:05:30Z",
  "episodes": [
    {
      "index": 1,
      "audio_file": "podcast_01.wav",
      "video_file": "podcast_01.mp4",
      "title": "Podcast Chapter 1",
      "source_chapters": [0],
      "duration_seconds": 333.0,
      "chapter_count": 1,
      "time_marks": [
        { "title": "Introducción", "seconds": 0.0,   "time": "0:00" },
        { "title": "Capítulo 1",   "seconds": 12.0,  "time": "0:12" },
        { "title": "Resumen",      "seconds": 334.0, "time": "5:34" },
        { "title": "Capítulo 1",   "seconds": 346.0, "time": "5:46" },
        { "title": "Despedida",    "seconds": 368.0, "time": "6:08" }
      ]
    },
    {
      "index": 2,
      "audio_file": "podcast_02.wav",
      "video_file": "podcast_02.mp4",
      "title": "Podcast Chapter 2",
      "source_chapters": [1, 2],
      "duration_seconds": 462.0,
      "chapter_count": 2,
      "time_marks": [
        { "title": "Introducción", "seconds": 0.0,   "time": "0:00" },
        { "title": "Capítulo 2",   "seconds": 12.0,  "time": "0:12" },
        { "title": "Capítulo 3",   "seconds": 373.0, "time": "6:13" },
        { "title": "Resumen",      "seconds": 465.0, "time": "7:45" },
        { "title": "Capítulo 2",   "seconds": 477.0, "time": "7:57" },
        { "title": "Capítulo 3",   "seconds": 509.0, "time": "8:29" },
        { "title": "Despedida",    "seconds": 521.0, "time": "8:41" }
      ]
    }
  ]
}
```

> Time-mark seconds above are illustrative — actual values depend on each
> audio clip's probed duration plus the 1s inter-segment silence after every
> clip. Capítulo 3 has no summary entry since its `summary` array is empty.

### `build()` return value (the result dict)

```json
{
  "script_id": "BookPodcastGenerator",
  "spec_path": "/abs/path/to/podcast.json",
  "started_at": "2026-07-01T12:00:00Z",
  "finished_at": "2026-07-01T12:05:30Z",
  "total_duration_seconds": 795.0,
  "sections": [
    { "name": "episode_1", "start_seconds": 0.0,   "end_seconds": 333.0, "duration_seconds": 333.0, "started_at": "...", "finished_at": "..." },
    { "name": "episode_2", "start_seconds": 333.0, "end_seconds": 795.0, "duration_seconds": 462.0, "started_at": "...", "finished_at": "..." }
  ],
  "outputs": [
    { "index": 0, "path": "/abs/output_folder/podcast_01.wav", "kind": "audio",    "label": "episode_1_audio", "section": "episode_1" },
    { "index": 1, "path": "/abs/output_folder/podcast_01.mp4", "kind": "video",    "label": "episode_1_video", "section": "episode_1" },
    { "index": 2, "path": "/abs/output_folder/podcast_02.wav", "kind": "audio",    "label": "episode_2_audio", "section": "episode_2" },
    { "index": 3, "path": "/abs/output_folder/podcast_02.mp4", "kind": "video",    "label": "episode_2_video", "section": "episode_2" },
    { "index": 4, "path": "/abs/output_folder/metadata.json",  "kind": "metadata", "label": "metadata" }
  ],
  "metadata": {
    "book_title": "Don Quijote",
    "author": "Cervantes",
    "intro_message": "Bienvenidos al podcast de Don Quijote",
    "episode_count": 2,
    "episodes": ["/* same as metadata.json episodes[] */"],
    "metadata_file": "/abs/output_folder/metadata.json"
  }
}
```

---

## Key points

- **Input** — spec JSON with `book_title`/`author`/`chapters` (required),
  plus optional `intro_message`/`welcome`/`summary_intro`/`farewell`. Each
  chapter's `content` is a non-empty array of audio files; `summary` is an
  optional array. No duration/silence knobs — those are internal.
- **Output** — one `podcast_NN.wav` + `podcast_NN.mp4` per episode, plus a
  run-level `metadata.json`. The MP4 video is identical across episodes
  except for total duration (3-phase: intro text 5s → title+author 5s →
  black for the remainder, audio throughout).
- **Time marks** — one per section (`welcome`, each chapter's content,
  `summary_intro`, each chapter's summary, `farewell`) — not per individual
  audio file within a chapter.