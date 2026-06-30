"""Build a slideshow video from images in a folder.

Core function: build_video(input_folder, display_seconds, transition_seconds)
CLI: python make_video.py <input_folder> [--display 5] [--transition 1]

Requires a movie.json file alongside the images:
    {
      "title": {"text": "My Video Title", "audio": "intro.mp3"},
      "images": [
        {"image": "0_134526.png", "text": "Caption for image 0", "audio": "0.mp3"},
        {"image": "1_134603.png", "text": "Caption for image 1", "audio": ""}
      ]
    }

`title` may be a string (no intro audio) or an object with `text` (required)
and `audio` (optional, resolved against the input folder). When `title.audio`
is present, the intro clip's duration equals that audio's duration and the
audio is attached to the intro; otherwise the intro is a silent 10s Voronoi
composition.

Each slide's duration equals its audio file's duration plus silent padding;
slides without audio use --display seconds. `transition_seconds` of silence is
inserted before and after every clip's audio so that consecutive audios never
overlap (the visual crossfade happens during the silent tails). `audio` is
optional; empty string or missing means no audio.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_DISPLAY = 5
DEFAULT_TRANSITION = 1
INTRO_SECONDS = 10
FPS = 30
CODEC = "libx264"
FONT_NAME = "DejaVuSans-Bold.ttf"
DRIFT_AMPLITUDE = 0.02  # fraction of canvas size
BORDER_WIDTH = 0.004   # fraction of min(w,h)
KENBURNS_OVERSAMPLE = 1.18  # pre-crop oversize factor vs shard bbox
KENBURNS_SCALE = 0.08   # max scale change over the intro (e.g. 1.0 -> 1.08)
KENBURNS_PAN = 0.06      # max pan fraction of the shard bbox


def natural_key(path: Path) -> list:
    s = path.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(input_folder: Path) -> list[Path]:
    files = [p for p in input_folder.iterdir()
             if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    files.sort(key=natural_key)
    return files


class MovieSpecError(ValueError):
    pass


def load_movie(input_folder: Path) -> tuple[dict, list[dict]]:
    """Load and validate movie.json. Returns (title_spec, images_entries).

    `title_spec` is a dict: {"text": <str>, "audio": <Path or None>}.

    Each entry: {"image": <name>, "text": <str or None>,
                 "audio": <Path or None>, "path": <Path>}.

    `image` is required. `text` and `audio` are optional. `audio` resolves
    against input_folder; empty/missing means no audio (uses fallback duration).

    `title` in movie.json may be a plain string (no intro audio) or an object
    {"text": <str>, "audio": <name>} where `audio` is optional and resolves
    against input_folder.
    """
    meta_path = input_folder / "movie.json"
    if not meta_path.is_file():
        raise MovieSpecError(f"required file missing: {meta_path}")

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MovieSpecError(f"invalid JSON in {meta_path}: {e}") from e

    if not isinstance(data, dict):
        raise MovieSpecError("movie.json top-level must be an object")

    raw_title = data.get("title")
    if raw_title is None:
        raise MovieSpecError("movie.json: 'title' is required")
    if isinstance(raw_title, str):
        if not raw_title.strip():
            raise MovieSpecError("movie.json: 'title' must be a non-empty string")
        title_text = raw_title
        title_audio_name: str = ""
    elif isinstance(raw_title, dict):
        title_text = raw_title.get("text")
        if not isinstance(title_text, str) or not title_text.strip():
            raise MovieSpecError(
                "movie.json: 'title.text' is required and must be a non-empty string")
        ta = raw_title.get("audio", "")
        if ta is None:
            ta = ""
        if not isinstance(ta, str):
            raise MovieSpecError("movie.json: 'title.audio' must be a string if present")
        title_audio_name = ta
    else:
        raise MovieSpecError(
            "movie.json: 'title' must be a string or an object {text, audio}")

    title_audio_path: Path | None = None
    if title_audio_name:
        title_audio_path = input_folder / title_audio_name
        if not title_audio_path.is_file():
            raise MovieSpecError(
                f"movie.json: title.audio '{title_audio_name}' not found in folder")

    title_spec = {"text": title_text, "audio": title_audio_path}

    entries = data.get("images")
    if not isinstance(entries, list) or len(entries) < 2:
        raise MovieSpecError("movie.json: 'images' must be a list with at least 2 entries")

    image_files = list_images(input_folder)
    image_names = {p.name: p for p in image_files}

    norm_entries: list[dict] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise MovieSpecError(f"movie.json: images[{i}] must be an object")
        fname = e.get("image")
        if not isinstance(fname, str) or not fname:
            raise MovieSpecError(f"movie.json: images[{i}].image is required")
        if fname not in image_names:
            raise MovieSpecError(
                f"movie.json: images[{i}].image '{fname}' not found in folder")
        text = e.get("text")
        if text is not None and not isinstance(text, str):
            raise MovieSpecError(f"movie.json: images[{i}].text must be a string if present")
        audio_name = e.get("audio", "")
        if audio_name is None:
            audio_name = ""
        if not isinstance(audio_name, str):
            raise MovieSpecError(f"movie.json: images[{i}].audio must be a string if present")
        audio_path: Path | None = None
        if audio_name:
            audio_path = (input_folder / audio_name)
            if not audio_path.is_file():
                raise MovieSpecError(
                    f"movie.json: images[{i}].audio '{audio_name}' not found in folder")
        norm_entries.append({
            "image": fname,
            "text": text,
            "audio": audio_path,
            "path": image_names[fname],
        })

    if len(norm_entries) != len(image_files):
        raise MovieSpecError(
            f"movie.json lists {len(norm_entries)} images but folder has "
            f"{len(image_files)} supported image files")

    return title_spec, norm_entries


def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_NAME, size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            sys.stderr.write(
                f"warning: could not load '{FONT_NAME}'; falling back to default font\n")
            return ImageFont.load_default()


def load_normalized(image_path: Path, target_w: int, target_h: int) -> np.ndarray:
    """Fit image into target_w x target_h preserving aspect, black-letterbox pad."""
    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return np.array(canvas)


def silent_audio_clip(duration: float, fps: int = 44100, nchannels: int = 2):
    """Return a silent AudioClip of the given duration."""
    from moviepy import AudioClip
    import numpy as _np

    if nchannels > 1:
        def make_frame(t):
            if isinstance(t, _np.ndarray):
                return _np.zeros((len(t), nchannels))
            return _np.zeros(nchannels)
    else:
        def make_frame(t):
            if isinstance(t, _np.ndarray):
                return _np.zeros(len(t))
            return 0.0

    return AudioClip(make_frame, duration=duration, fps=fps)


def padded_audio(audio_clip, pad_seconds: float):
    """Wrap `audio_clip` in `pad_seconds` of silence on both sides.

    Returns a CompositeAudioClip of total duration `audio_clip.duration +
    2*pad_seconds` with the original audio starting at `pad_seconds` and
    `pad_seconds` of silence after it ends.
    """
    from moviepy import CompositeAudioClip

    fps = audio_clip.fps or 44100
    nch = audio_clip.nchannels if hasattr(audio_clip, "nchannels") else 2
    pad_before = silent_audio_clip(pad_seconds, fps=fps, nchannels=nch)
    pad_after = silent_audio_clip(pad_seconds, fps=fps, nchannels=nch)
    audio_dur = float(audio_clip.duration)
    return CompositeAudioClip([
        pad_before,
        audio_clip.with_start(pad_seconds),
        pad_after.with_start(pad_seconds + audio_dur),
    ])


def render_title_overlay(size: tuple[int, int], title: str) -> Image.Image:
    """Render a centered title with outlined text (same style as captions),
    auto-fit to 90% of canvas width, with a 2-line wrap fallback. Never off-screen."""
    w, h = size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    max_w = int(w * 0.9)
    min_size = max(20, int(h * 0.04))
    start_size = max(min_size, int(h * 0.12))

    def fit_size(text: str, font_size: int) -> tuple[int, tuple[int, int, int, int]] | None:
        font = load_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        if tw <= max_w:
            return font_size, bbox
        return None

    # Try single line at decreasing sizes.
    chosen = None
    for size in range(start_size, min_size - 1, -2):
        res = fit_size(title, size)
        if res:
            chosen = res
            break

    lines = [title]
    if not chosen:
        # Wrap into two lines at the min size (split near the middle space).
        words = title.split()
        if len(words) >= 2:
            mid = len(words) // 2
            line1 = " ".join(words[:mid])
            line2 = " ".join(words[mid:])
        else:
            line1, line2 = title, ""
        # Fit each line at decreasing size.
        for size in range(min_size + 8, min_size - 1, -1):
            f1 = load_font(size)
            b1 = draw.textbbox((0, 0), line1, font=f1)
            b2 = draw.textbbox((0, 0), line2, font=f1) if line2 else b1
            if (b1[2] - b1[0]) <= max_w and (b2[2] - b2[0]) <= max_w:
                chosen = (size, (b1, b2))
                break
        lines = [line1, line2]

    if not chosen:
        # Last resort: min size, single line (may exceed width but stays centered).
        chosen = (min_size, draw.textbbox((0, 0), title, font=load_font(min_size)))
        lines = [title]

    font_size = chosen[0]
    font = load_font(font_size)
    outline_w = max(3, int(h * 0.015))

    # Compute vertical placement for 1 or 2 lines so the block is centered.
    bboxes = []
    for ln in lines:
        if ln:
            bboxes.append(draw.textbbox((0, 0), ln, font=font))
    total_h = sum(b[3] - b[1] for b in bboxes) + (len(bboxes) - 1) * int(h * 0.02)
    y_cursor = (h - total_h) // 2
    for ln, bb in zip(lines, bboxes):
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        tx = (w - tw) // 2 - bb[0]
        ty = y_cursor - bb[1]
        draw.text((tx, ty), ln, fill=(255, 255, 255, 255), font=font,
                  stroke_width=outline_w, stroke_fill=(0, 0, 0, 255))
        y_cursor += th + int(h * 0.02)
    return layer


def polygon_area(poly: list[tuple[int, int]]) -> float:
    """Shoelace area of a polygon (list of (x,y))."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def balanced_shards(w: int, h: int, rng: np.random.Generator) -> list[list[tuple[int, int]]]:
    """Return 3 shards (2 triangles + 1 quadrilateral) tiling [0,w]x[0,h].

    Pick an interior point P near the center; connect P to all 4 corners,
    forming 4 triangles. Merge the 2 smallest (adjacent, sharing the P-corner
    edge) into a quadrilateral. Result: 3 shards, each ~33% of the canvas.
    """
    px = int(rng.uniform(0.35 * w, 0.65 * w))
    py = int(rng.uniform(0.35 * h, 0.65 * h))
    P = (px, py)
    A, B, C, D = (0, 0), (w, 0), (w, h), (0, h)
    tris = [
        ([A, B, P], 0),  # top
        ([B, C, P], 1),  # right
        ([C, D, P], 2),  # bottom
        ([D, A, P], 3),  # left
    ]
    # Sort by area; the two smallest are merged.
    tris.sort(key=lambda t: polygon_area(t[0]))
    small1, small2 = tris[0], tris[1]
    big = tris[2][0], tris[3][0]

    # Merge the two smallest triangles. They share the edge from P to their
    # common corner. Determine the shared corner and build a quadrilateral
    # by concatenating the non-shared vertices.
    # Each triangle is [cornerA, cornerB, P]; the shared corner is the one
    # that appears in both (besides P).
    c1 = [v for v in small1[0] if v != P]
    c2 = [v for v in small2[0] if v != P]
    # The two triangles share P and one corner (they are adjacent wedges).
    shared_corner = set(c1) & set(c2)
    if len(shared_corner) != 1:
        # Fallback: just concatenate (shouldn't happen with 4 consecutive wedges).
        merged = small1[0] + [v for v in small2[0] if v != P]
    else:
        shared = list(shared_corner)[0]
        # Build quad: go around shared corner -> P -> outer vertices.
        # Take the two non-shared corners (one from each triangle).
        outer1 = [v for v in c1 if v != shared][0]
        outer2 = [v for v in c2 if v != shared][0]
        # Order: shared, outer1, P, outer2 (or any convex order).
        merged = [shared, outer1, P, outer2]

    shards = [big[0], big[1], merged]
    return shards


def build_composition_clip(
    image_paths: list[Path],
    title: dict,
    target_w: int,
    target_h: int,
    duration: float,
    rng: np.random.Generator,
):
    """Build the opening composition: 3 balanced shards with Ken Burns
    animation, black gradient borders, and the title overlay.

    `title` is a dict {"text": <str>, "audio": <Path or None>}; only `text` is
    used here (audio is attached by the caller). `duration` controls the clip
    length and the Ken Burns progress interpolation."""
    from moviepy import VideoClip

    shards = balanced_shards(target_w, target_h, rng)
    if not shards:
        raise RuntimeError("shard generation produced no shards")

    # Pre-load source images as PIL RGB.
    sources = [Image.open(p).convert("RGB") for p in image_paths]

    border_w = max(3, int(BORDER_WIDTH * min(target_w, target_h)))

    # For each shard, pick a source image and pre-crop an oversized region
    # (KENBURNS_OVERSAMPLE x the shard bbox) so Ken Burns pan/zoom never
    # reveals empty edges. Randomize scale direction and pan vector.
    shard_data = []
    for i, poly in enumerate(shards):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bw = max(1, max(xs) - min(xs))
        bh = max(1, max(ys) - min(ys))
        src = sources[i % len(sources)]
        sw, sh = src.size

        # Oversized crop matching shard bbox aspect.
        over_w = int(round(bw * KENBURNS_OVERSAMPLE))
        over_h = int(round(bh * KENBURNS_OVERSAMPLE))
        over_w = min(over_w, sw)
        over_h = min(over_h, sh)
        # If source is smaller than the oversize in one dimension, match aspect.
        if sw / sh > over_w / over_h:
            over_w = int(round(over_h * sw / sh))
            over_w = min(over_w, sw)
        else:
            over_h = int(round(over_w * sh / sw))
            over_h = min(over_h, sh)
        cx = int(rng.integers(0, max(1, sw - over_w + 1)))
        cy = int(rng.integers(0, max(1, sh - over_h + 1)))
        pre_crop = src.crop((cx, cy, cx + over_w, cy + over_h))

        # Ken Burns params: scale start/end and pan start/end (in pre_crop px).
        scale0 = 1.0
        scale1 = 1.0 + float(rng.uniform(0.5, 1.0) * KENBURNS_SCALE)
        # Pan range as fraction of the oversize margin.
        margin_x = pre_crop.size[0] - bw
        margin_y = pre_crop.size[1] - bh
        pan_x0 = float(rng.uniform(0, max(0, margin_x)))
        pan_y0 = float(rng.uniform(0, max(0, margin_y)))
        pan_x1 = float(rng.uniform(0, max(0, margin_x)))
        pan_y1 = float(rng.uniform(0, max(0, margin_y)))

        shard_data.append({
            "poly": poly,
            "min_x": min(xs), "min_y": min(ys),
            "bw": bw, "bh": bh,
            "pre_crop": pre_crop,
            "scale0": scale0, "scale1": scale1,
            "pan_x0": pan_x0, "pan_y0": pan_y0,
            "pan_x1": pan_x1, "pan_y1": pan_y1,
        })

    title_alpha = render_title_overlay((target_w, target_h), title["text"])

    def make_frame(t: float) -> np.ndarray:
        progress = (t / duration) if duration > 0 else 0.0
        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 255))
        for sd in shard_data:
            scale = sd["scale0"] + (sd["scale1"] - sd["scale0"]) * progress
            pan_x = sd["pan_x0"] + (sd["pan_x1"] - sd["pan_x0"]) * progress
            pan_y = sd["pan_y0"] + (sd["pan_y1"] - sd["pan_y0"]) * progress
            # Sub-crop from pre_crop at the current scale/pan, sized to shard bbox.
            sub_w = int(round(sd["bw"] / scale))
            sub_h = int(round(sd["bh"] / scale))
            sub_w = min(sub_w, sd["pre_crop"].size[0])
            sub_h = min(sub_h, sd["pre_crop"].size[1])
            sx = int(round(pan_x))
            sy = int(round(pan_y))
            sx = min(max(sx, 0), max(0, sd["pre_crop"].size[0] - sub_w))
            sy = min(max(sy, 0), max(0, sd["pre_crop"].size[1] - sub_h))
            sub = sd["pre_crop"].crop((sx, sy, sx + sub_w, sy + sub_h))
            frame_img = sub.resize((sd["bw"], sd["bh"]), Image.LANCZOS)

            mask = Image.new("L", (sd["bw"], sd["bh"]), 0)
            ImageDraw.Draw(mask).polygon(
                [(p[0] - sd["min_x"], p[1] - sd["min_y"]) for p in sd["poly"]],
                fill=255)
            canvas.paste(frame_img.convert("RGBA"),
                         (sd["min_x"], sd["min_y"]), mask)

        # Black gradient borders on each shard.
        border_layer = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(border_layer)
        for sd in shard_data:
            pts = sd["poly"]
            bdraw.line(pts + [pts[0]], fill=(0, 0, 0, 255), width=border_w)
        # Slight blur for gradient falloff.
        border_layer = border_layer.filter(ImageFilter.GaussianBlur(border_w // 2 + 1))
        canvas = Image.alpha_composite(canvas, border_layer)

        # Composite title (with its own alpha) on top.
        out = Image.alpha_composite(canvas, title_alpha)
        return np.array(out.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    return clip


def render_caption_overlay(size: tuple[int, int], text: str) -> Image.Image:
    """Render a centered caption with outlined text (no background) on an RGBA layer."""
    w, h = size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = load_font(max(20, int(h * 0.06)))

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    tx = (w - tw) // 2 - bbox[0]
    ty = (h - th) // 2 - bbox[1]
    outline_w = max(2, int(h * 0.012))
    draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font,
              stroke_width=outline_w, stroke_fill=(0, 0, 0, 255))
    return layer


def overlay_caption_on_array(frame_arr: np.ndarray, text: str) -> np.ndarray:
    """Composite a centered caption with scrim onto an RGB numpy frame."""
    h, w = frame_arr.shape[:2]
    base = Image.fromarray(frame_arr).convert("RGBA")
    caption = render_caption_overlay((w, h), text)
    out = Image.alpha_composite(base, caption)
    return np.array(out.convert("RGB"))


def build_video(
    input_folder: str | os.PathLike,
    display_seconds: float = DEFAULT_DISPLAY,
    transition_seconds: float = DEFAULT_TRANSITION,
) -> dict:
    """Build a slideshow video with an opening composition + captioned images.

    The intro clip's duration is the title audio's duration (if present) plus
    `transition_seconds` of silent padding on each side; otherwise it falls
    back to INTRO_SECONDS. Each slide's clip duration is its audio duration
    (or --display when no audio) plus `transition_seconds` of silence on each
    side, so consecutive audios are separated by `transition_seconds` of clean
    silence and never overlap (the visual crossfade happens during the silent
    tails).

    Returns a dict with:
      video_path:  relative path "output/<timestamp>.mp4"
      metadata:    {...}
    """
    input_folder = Path(input_folder).resolve()
    if not input_folder.is_dir():
        raise FileNotFoundError(f"input_folder does not exist: {input_folder}")

    title, entries = load_movie(input_folder)
    image_paths = [e["path"] for e in entries]

    from moviepy import ImageClip, CompositeVideoClip, AudioFileClip
    try:
        from moviepy.video.fx import CrossFadeIn
    except Exception:
        from moviepy.video.fx.all import CrossFadeIn  # type: ignore

    # Target size from the first image.
    with Image.open(image_paths[0]) as im:
        target_w, target_h = im.size

    rng = np.random.default_rng(1234)

    # 1. Intro composition clip.
    #    Duration = title_audio.duration + 2*transition (silence-padded) when
    #    title audio is present; otherwise the silent INTRO_SECONDS default.
    intro_audio_clips: list = []
    if title.get("audio") is not None:
        title_audio = AudioFileClip(str(title["audio"]))
        intro_audio_clips.append(title_audio)
        intro_padded_audio = padded_audio(title_audio, transition_seconds)
        intro_duration = float(intro_padded_audio.duration)
        print(f"[movie-py] Building intro composition ({intro_duration:.1f}s, "
              f"title audio {title_audio.duration:.1f}s) ...", flush=True)
    else:
        intro_padded_audio = None
        intro_duration = float(INTRO_SECONDS)
        print(f"[movie-py] Building intro composition ({intro_duration:.1f}s, "
              f"silent) ...", flush=True)

    intro_clip = build_composition_clip(
        image_paths, title, target_w, target_h, intro_duration, rng)
    if intro_padded_audio is not None:
        intro_clip = intro_clip.with_audio(intro_padded_audio)

    # 2. Per-image clips with captions and silence-padded audio.
    #    Clip duration = audio duration (or --display) + 2*transition_seconds.
    from tqdm import tqdm

    image_clips: list = []
    durations: list[float] = []
    audio_clips: list = []
    pad = float(transition_seconds)
    for e in tqdm(entries, desc="Preparing slides", unit="slide"):
        arr = load_normalized(e["path"], target_w, target_h)
        text = e.get("text")
        if text:
            arr = overlay_caption_on_array(arr, text)
        if e.get("audio") is not None:
            a = AudioFileClip(str(e["audio"]))
            audio_clips.append(a)
            padded = padded_audio(a, pad)
            dur = float(padded.duration)
            clip = ImageClip(arr).with_duration(dur).with_audio(padded)
        else:
            dur = float(display_seconds) + 2 * pad
            clip = ImageClip(arr).with_duration(dur)
        image_clips.append(clip)
        durations.append(dur)

    # Assemble timeline: intro + image_clips with crossfades.
    # intro starts at 0; first image starts at (intro_duration - transition) so
    # it crossfades over the tail of the intro; subsequent images step by
    # (prev_duration - transition_seconds).
    #
    # Because each clip's duration includes 2*transition of silent padding
    # around its audio, the next clip's audio starts exactly transition_seconds
    # after the previous clip's audio ends -- no audio overlap, and the visual
    # crossfade happens entirely within the silent tails.
    intro_clip = intro_clip.with_start(0)
    start = intro_duration - transition_seconds
    for i, clip in enumerate(image_clips):
        clip = clip.with_start(start)
        clip = clip.with_effects([CrossFadeIn(transition_seconds)])
        image_clips[i] = clip
        start += durations[i] - transition_seconds

    # After the loop, `start` is one transition past the last clip's end
    # (it was incremented after the last clip was placed). Add it back.
    total_duration = start + transition_seconds

    composite = CompositeVideoClip(
        [intro_clip] + image_clips, size=(target_w, target_h))
    composite = composite.with_duration(total_duration)

    # Output: <input>/output/<timestamp>.mp4
    output_dir = input_folder / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    video_filename = f"{timestamp}.mp4"
    output_path = output_dir / video_filename

    composite.write_videofile(
        str(output_path),
        fps=FPS,
        codec=CODEC,
        audio_codec="aac",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
        logger="bar",
    )

    for c in [intro_clip] + image_clips + audio_clips + intro_audio_clips:
        try:
            c.close()
        except Exception:
            pass
    composite.close()

    file_size = output_path.stat().st_size
    frame_count = round(total_duration * FPS)

    print(
        f"[movie-py] Video: {total_duration:.1f}s, {len(image_clips)} slides, "
        f"{target_w}x{target_h}, {file_size/1e6:.1f}MB -> {video_filename}",
        flush=True,
    )

    return {
        "video_path": f"output/{video_filename}",
        "metadata": {
            "duration_seconds": round(total_duration, 3),
            "frame_count": frame_count,
            "fps": FPS,
            "resolution": f"{target_w}x{target_h}",
            "image_count": len(image_clips),
            "intro_seconds": round(intro_duration, 3),
            "title": title["text"],
            "title_audio": title["audio"].name if title.get("audio") else None,
            "transition_seconds": transition_seconds,
            "display_seconds": display_seconds,
            "codec": "h264",
            "file_size_bytes": file_size,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a slideshow video from images.")
    parser.add_argument("input_folder", help="Folder containing input images + movie.json")
    parser.add_argument("--display", type=float, default=DEFAULT_DISPLAY,
                        help="Seconds each slide displays when it has no audio (default 5)")
    parser.add_argument("--transition", type=float, default=DEFAULT_TRANSITION,
                        help="Crossfade duration in seconds (default 1)")
    args = parser.parse_args(argv)

    try:
        result = build_video(args.input_folder, args.display, args.transition)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())