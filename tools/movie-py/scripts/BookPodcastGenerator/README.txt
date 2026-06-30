BookPodcastGenerator

Builds a multi-episode book podcast from a spec describing full-chapter audio
files plus book-level welcome / summary intro / farewell audio.

Spec (podcast.json-style; audio filenames resolve against input_folder):
    {
      "book_title": "Don Quijote",                       // required
      "author": "Cervantes",                            // required
      "min_chapter_duration_seconds": 300,              // optional, default 300
      "target_chapter_duration_seconds": 600,           // optional, default 600
      "silence_between_segments_ms": 1000,              // optional, default 1000
      "trailing_silence_ms": 5000,                       // optional, default 5000
      "welcome":       "welcome.wav",                   // optional, book-level
      "summary_intro": "summary_intro.wav",             // optional, book-level
      "farewell":      "farewell.wav",                   // optional, book-level
      "chapters": [                                      // required, non-empty
        {"title": "Capitulo 1", "content": "ch01.wav", "summary": "sum01.wav"},
        {"title": "Capitulo 2", "content": "ch02.wav", "summary": "sum02.wav"}
      ]
    }

Required:
  - book_title, author, chapters (non-empty array).
Optional (with built-in defaults):
  - min_chapter_duration_seconds (default 300)
  - target_chapter_duration_seconds (default 600)
  - silence_between_segments_ms (default 1000)
  - trailing_silence_ms (default 5000)
  - welcome, summary_intro, farewell (book-level audio; omitted = no such
    segment in any episode).

Per chapter: `content` is required; `summary` is optional.

Behavior:
  - Chapters are grouped into episodes by content duration: a new episode
    starts when adding the next chapter would exceed target_chapter_duration_seconds.
    Undersized episodes (< min_chapter_duration_seconds) are merged forward
    (non-last) or backward (last).
  - Each episode concatenates, in order: welcome -> each grouped chapter's
    content -> summary intro -> each grouped chapter's summary (if any) ->
    farewell. silence_between_segments_ms is inserted after every segment;
    trailing_silence_ms is appended once at the end of each episode.
  - Per-segment time marks are recorded as cumulative seconds at the segment
    start (before appending it), formatted as "M:SS".

Outputs (written to output_folder):
  - podcast_01.wav, podcast_02.wav, ... (16-bit PCM stereo)
  - podcast_01.mp4, podcast_02.mp4, ... (1920x1080 near-black background with
    faded title/author overlay, AAC audio, h264/yuv420p, 30fps)
  - metadata.json (run-level: book_title, author, started_at, finished_at,
    episodes[] each with index/audio_file/video_file/title/source_chapters/
    duration_seconds/chapter_count/time_marks)

Params (override spec values; validated against PARAM_SCHEMA; all optional,
default to the spec value or the built-in default above):
  min_chapter_duration_seconds, target_chapter_duration_seconds,
  silence_between_segments_ms, trailing_silence_ms  (numbers >= 0)

Result: section-aware dict per the movie-py contract (one section per
episode), with metadata.episodes carrying the full per-episode metadata
and metadata.metadata_file pointing at the written metadata.json.