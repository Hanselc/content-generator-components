Scripts registry for movie-py.

Layout - one subfolder per script, keyed by scriptId:

  scripts/
    <scriptId>/
      script.py            # required: build(ctx) -> dict, optional PARAM_SCHEMA, META
      README.txt           # optional, script-specific notes
    README.txt             # this file

The scriptId is whatever the subfolder is named (e.g. SocialMediaGenerator).
There is no default script: every request must name a scriptId that matches a
preexisting subfolder here, otherwise the server returns 400.

Module contract
---------------
Each script.py must expose:

  build(ctx) -> dict        (required)
      Performs the movie construction and returns a section-aware result dict
      (see "Result shape" below).

  PARAM_SCHEMA: dict        (optional)
      A JSON-Schema-shaped dict describing the expected shape of the request's
      `params` object. When present, the server validates `params` against it
      BEFORE calling build() and returns 400 on mismatch. When absent, `params`
      is passed through unvalidated. Supported schema keywords: type, properties,
      required, additionalProperties, minimum, maximum, enum, items, default.

  META: dict                (optional)
      {"description": str, "version": str} surfaced by GET /scripts/<scriptId>
      for callers/UIs that want to describe the script.

The ctx object
--------------
build(ctx) receives a SimpleNamespace with:

  ctx.params         dict          the dynamic per-script payload (validated)
  ctx.input_folder   Path          absolute, verified to exist (image/audio assets)
  ctx.output_folder  Path          absolute, created by the server if missing
  ctx.spec_path      Path          absolute, verified to exist (the spec file
                                   describing what to build, e.g. movie.json)
  ctx.common         dict          {"script_id": str, "started_at": ISO8601 str}
  ctx.primitives     module        the make_video module, exposing the shared
                                   rendering/composition helpers so scripts
                                   reuse them instead of reimplementing:

        load_movie(spec_path, input_folder) -> (title_spec, entries)
        list_images(input_folder) -> [Path]
        load_normalized(image_path, target_w, target_h) -> np.ndarray
        padded_audio(audio_clip, pad_seconds) -> CompositeAudioClip
        silent_audio_clip(duration, fps=44100, nchannels=2) -> AudioClip
        build_composition_clip(image_paths, title, w, h, duration, rng) -> VideoClip
        render_caption_overlay(size, text) -> Image.Image
        render_title_overlay(size, title) -> Image.Image
        overlay_caption_on_array(frame_arr, text) -> np.ndarray
        balanced_shards(w, h, rng) -> [polygon]
        load_font(size) -> ImageFont.FreeTypeFont
        assemble(...) -> dict      (the finalizer: writes the .mp4 and returns
                                    the section-aware result dict with time marks)

        Constants: FPS, CODEC, DEFAULT_DISPLAY, DEFAULT_TRANSITION, INTRO_SECONDS,
                   SUPPORTED_EXT, FONT_NAME, DRIFT_AMPLITUDE, BORDER_WIDTH,
                   KENBURNS_OVERSAMPLE, KENBURNS_SCALE, KENBURNS_PAN
        Errors:    MovieSpecError(ValueError)

Deduplication
-------------
Scripts MUST reuse the primitives in ctx.primitives (or `import make_video`)
rather than reimplementing padded_audio, build_composition_clip, etc. This
keeps the rendering logic in one place.

Result shape
------------
build() must return a dict shaped like:

  {
    "script_id":              "<scriptId>",
    "spec_path":              "/abs/path/to/spec.json" | null,
    "started_at":             "2026-06-30T11:30:00Z",
    "finished_at":            "2026-06-30T11:30:42Z",
    "total_duration_seconds": 38.5,
    "sections": [
      {"name": "intro",  "start_seconds": 0.0,  "end_seconds": 11.0,
       "duration_seconds": 11.0, "started_at": "...", "finished_at": "..."},
      ...
    ],
    "outputs": [
      {"index": 0, "path": "/abs/path/to/<file>.<ext>", "kind": "video",
       "label": "main", "section": "intro"},
      ...
    ],
    "metadata": { ... script-specific + common fields ... }
  }

`outputs` is REQUIRED and is a non-empty list of output-file descriptors, one
per file produced. The server validates it after build() runs. Each entry:
  - index:   int, required, unique, 0-based within the list.
  - path:    str, required, absolute; must exist on disk and resolve inside
             ctx.output_folder.
  - kind:    str, required, one of: "audio", "video", "metadata", "image",
             "other".
  - label:   str, optional, a script-defined stable identifier
             (e.g. "episode_1_video").
  - section: str, optional, the name of a section in `sections` this output
             belongs to.

There is no top-level `video_path` field (removed from the contract). A
script producing a single video simply returns `outputs` with one entry of
kind "video".

The simplest path is to call ctx.primitives.assemble(...), which writes the
composite to <output_folder>/<timestamp>.mp4 and constructs this dict for you
from the sections list and clips you hand it; it returns a single
`outputs` entry (index 0, kind "video", label "main").

Request shape (POST /generate)
------------------------------
  {
    "scriptId":     "<scriptId>",        # required, must preexist here
    "input_folder": "/abs/...",          # required, absolute
    "output_folder":"/abs/...",          # required, absolute
    "spec_path":    "/abs/...",          # required, absolute
    "params":       { ... }              # dynamic, validated against PARAM_SCHEMA
  }

To add a new script:
  1. Create scripts/<scriptId>/ with a script.py exposing build(ctx) -> dict.
  2. Optionally declare PARAM_SCHEMA (recommended: lets the server reject
     malformed requests with a clear 400 before build() runs) and META.
  3. No restart is strictly needed for discovery (scripts are loaded per
     request), but if you change a script's code during a long-running server
     process, the next request picks up the new code automatically.