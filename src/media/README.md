# media/

Video, audio and single-image understanding: YouTube and local media in, transcripts, descriptions
and audio files out.

| File | What it does |
| --- | --- |
| `video.py` | `/video <link or file> [question]`: downloads/reads a video, transcribes it, looks at frames with the local vision model. Owns the FFmpeg and yt-dlp setup. |
| `see.py` | `/see <file or url> [question]`: describes one image with the same local vision model as `/video`. |
| `music.py` | Music downloader: YouTube / YouTube Music songs and playlists → tagged audio files. |
| `music_followup.py` | Follow-ups in normal chat about songs a music job skipped ("download them anyway", "download #15"). |

**Why they're together:** `video.py` is the shared base. The others reuse its pieces instead of
setting them up again:

- `see.py` uses `video.Vision` (the Qwen2.5-VL vision model wrapper) and `video.quiet`.
- `music.py` uses `video.ydl_opts` (shared yt-dlp options) and `video.fmt_time`.

`music.py` ↔ `music_followup.py` is a circular pair. `music.py` imports the follow-up module **lazily**,
inside functions. Keep it that way, or the import fails.

Outside users: `chat.py`, `photos/faces_followup.py` (reuses `music`'s text-matching helpers) and
`photos/layer_models.py` (borrows `video.Vision` for the optional description).
