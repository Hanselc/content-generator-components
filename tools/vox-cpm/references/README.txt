Reference audio files for VoxCPM Hi-Fi Cloning.

Files:
  reference.wav              - prompt clip (paired with prompt_text in reference.json)
  reference_reinforce.wav    - reinforce clip for extra timbre (used when
                               reinforce_enabled is true in reference.json)

Guidelines for low-VRAM GPUs (8 GB):
  - Keep reference clips to 5-8 seconds. Longer clips OOM the AudioVAE
    encoder which processes the entire clip in float32 in one pass.
  - Truncate at a silence boundary (word or sentence complete), NOT at an
    arbitrary timestamp. Cutting mid-word corrupts both the audio and the
    transcript match required by Hi-Fi cloning.
  - The prompt_text field in reference.json MUST be the exact transcript of
    the (truncated) prompt clip. If you shorten the audio, update prompt_text
    to match. Use ASR (e.g. SenseVoice, Whisper) to obtain the transcript
    if you are unsure of the exact words.

To re-enable reinforcement on a GPU with more VRAM:
  1. Truncate reference_reinforce.wav to 5-8s at a silence boundary.
  2. Set "reinforce_enabled": true in reference.json.
  3. Restart the server.