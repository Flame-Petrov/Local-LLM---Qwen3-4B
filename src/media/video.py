"""
Video understanding: "/video <link or file> [question]" or just paste a video link.

Pipeline (everything runs locally; results are cached per video in downloads/videos/<id>/):
  1. metadata    yt-dlp: title, channel, date, length, views, chapters, description, top comments
  2. speech      the video's own captions (manual first, then auto) - or, if there are none,
                 the audio is transcribed locally with Whisper (faster-whisper)
  3. visuals     FFmpeg samples frames, the most distinct frame of each time slice is kept,
                 and a local vision model (Qwen2.5-VL) describes it + reads on-screen text
  4. dossier     all of it as one timestamped text; if it is too long for the chat model,
                 the transcript is condensed into timestamped notes first
The dossier goes into the chat history, so the user can keep asking about the video
(and /write it into a file). Settings and prompts: config.json -> "video".
"""

import base64
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from core import config
from core.config import text
from core.debuglog import log

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".3gp")


# ---------------- request ----------------

def parse_request(message: str, vcfg: dict):
    """(source, question) if the message is about a video, else None.
    '/video <url|path> [question]', or any message containing a video-site link or a local video file."""
    msg = message.strip()
    is_cmd = msg.lower().startswith("/video")
    body = msg[len("/video"):].strip() if is_cmd else msg
    # local file (quoted or not)
    for cand in re.findall(r'"([^"]+)"', body) + body.split():
        p = Path(cand.strip("'\""))
        if p.suffix.lower() in VIDEO_EXTS and p.is_file():
            return str(p.resolve()), body.replace(cand, " ").replace('""', " ").strip()
    url = re.search(vcfg["url_pattern"], body)
    if url and (is_cmd or re.search(vcfg["video_url_pattern"], url.group(0), re.I)):
        src = url.group(0).rstrip(".,;:!?)\"'")
        return src, (body[: url.start()] + " " + body[url.end():]).strip()
    return None


def fmt_time(sec) -> str:
    sec = int(sec or 0)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ---------------- tools ----------------

def ffmpeg_exe(vcfg) -> str:
    """ffmpeg from PATH, else the copy bundled with imageio-ffmpeg (copied as tools/ffmpeg.exe,
    the name yt-dlp looks for)."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    tools = config.path(vcfg["tools_dir"])
    exe = tools / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not exe.exists():
        import imageio_ffmpeg
        tools.mkdir(parents=True, exist_ok=True)
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), exe)
    return str(exe)


def ydl_opts(vcfg, **extra):
    o = {"quiet": True, "no_warnings": True, "noprogress": True, "ffmpeg_location": ffmpeg_exe(vcfg),
         "socket_timeout": vcfg["timeout_s"], "retries": 3}
    if vcfg["js_runtime"]:
        o["js_runtimes"] = {vcfg["js_runtime"]: {}}
    if vcfg["cookies_from_browser"]:  # for sites that need a login (Instagram, TikTok, X...)
        o["cookiesfrombrowser"] = (vcfg["cookies_from_browser"],)
    o.update(extra)
    return o


def media_duration(path, vcfg) -> float:
    try:
        r = subprocess.run([ffmpeg_exe(vcfg), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                           encoding="utf-8", errors="ignore", timeout=60)
    except subprocess.TimeoutExpired:
        return 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


# ---------------- 1. metadata ----------------

def get_info(src, vcfg, folder):
    cache = folder / "info.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if Path(src).is_file():
        p = Path(src)
        info = {"title": p.stem, "duration": media_duration(p, vcfg), "local_file": str(p), "webpage_url": str(p)}
    else:
        import yt_dlp
        n = str(vcfg["max_comments"])
        opts = ydl_opts(vcfg, skip_download=True, getcomments=vcfg["max_comments"] > 0,
                        extractor_args={"youtube": {"max_comments": [n, n, "0", "0"], "comment_sort": ["top"]}})
        with yt_dlp.YoutubeDL(opts) as y:
            raw = y.extract_info(src, download=False)
            if raw.get("_type") == "playlist" and raw.get("entries"):
                raw = next(e for e in raw["entries"] if e)
            raw = y.sanitize_info(raw)
        keep = ("id", "title", "uploader", "channel", "upload_date", "duration", "view_count", "like_count",
                "comment_count", "description", "chapters", "tags", "categories", "language", "webpage_url",
                "extractor_key", "subtitles", "automatic_captions", "is_live")
        info = {k: raw.get(k) for k in keep}
        info["comments"] = [{"text": c.get("text", ""), "likes": c.get("like_count") or 0, "author": c.get("author")}
                            for c in (raw.get("comments") or [])][: vcfg["max_comments"]]
    cache.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


# ---------------- 2. speech ----------------

def _parse_vtt(data: str):
    """WebVTT -> [(start, text)]; removes the rolling duplicates of YouTube auto-captions."""
    segs, seen = [], {}  # dict: keeps insertion order, so trimming keeps the most recent lines
    for block in re.split(r"\n\s*\n", data):
        m = re.search(r"(\d+:)?(\d+):(\d+)[.,](\d+)\s*-->", block)
        if not m:
            continue
        start = int(m.group(1)[:-1] if m.group(1) else 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        lines = block[m.end():].split("\n")[1:]
        for line in lines:
            t = re.sub(r"<[^>]+>", "", line).strip()
            t = re.sub(r"\s+", " ", t.replace("&nbsp;", " ").replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<"))
            if t and t not in seen:
                seen[t] = None
                segs.append((start, t))
        if len(seen) > 4000:
            seen = dict.fromkeys(list(seen)[-200:])
    return segs


def _parse_json3(data: str):
    segs = []
    for ev in json.loads(data).get("events", []):
        t = "".join(s.get("utf8", "") for s in ev.get("segs") or []).replace("\n", " ").strip()
        if t:
            segs.append((ev.get("tStartMs", 0) / 1000, t))
    return segs


def _pick_track(info, vcfg):
    """Best caption track: manual in the video's language / preferred languages, then auto-captions."""
    lang = (info.get("language") or "").split("-")[0]
    prefs = [x for x in [lang] + vcfg["caption_languages"] if x]
    for kind, tracks in (("manual", info.get("subtitles") or {}), ("auto", info.get("automatic_captions") or {})):
        tracks = {k: v for k, v in tracks.items() if k != "live_chat"}
        if not tracks:
            continue
        order = [f"{p}-orig" for p in prefs] + prefs if kind == "auto" else prefs
        # auto-captions list every translation; '<lang>-orig' marks the spoken language
        orig = [k for k in tracks if k.endswith("-orig")]
        keys = [k for k in order if k in tracks] + orig + [k for k in tracks if k.split("-")[0] in prefs]
        if kind == "manual":
            keys += list(tracks)
        for k in keys:
            fmts = {f.get("ext"): f.get("url") for f in tracks[k] if f.get("url")}
            for ext in ("json3", "vtt"):
                if fmts.get(ext):
                    return kind, k, ext, fmts[ext]
    return None


def get_captions(info, vcfg, folder):
    track = _pick_track(info, vcfg)
    if not track:
        return None
    kind, lang, ext, url = track
    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_opts(vcfg)) as y:
        data = y.urlopen(url).read().decode("utf-8", "ignore")
    segs = _parse_json3(data) if ext == "json3" else _parse_vtt(data)
    if not segs:
        return None
    return {"source": f"{kind} captions ({lang})", "language": lang.replace("-orig", ""), "segments": segs}


def download(src, vcfg, folder, what):
    """Download audio or a small video stream (no merging needed). Returns the file path."""
    if Path(src).is_file():
        return Path(src)
    import yt_dlp
    fmt = vcfg["audio_format"] if what == "audio" else vcfg["video_format"]
    with yt_dlp.YoutubeDL(ydl_opts(vcfg, format=fmt, outtmpl=str(folder / f"{what}.%(ext)s"))) as y:
        y.extract_info(src, download=True)
    files = [p for p in folder.glob(f"{what}.*") if not p.name.endswith((".part", ".ytdl"))]
    if not files:
        raise RuntimeError(f"{what} download produced no file")
    return files[0]


def transcribe(media: Path, vcfg, duration, progress):
    from faster_whisper import WhisperModel
    w = vcfg["whisper"]
    progress(f"loading speech model ({w['model']})")
    model = WhisperModel(w["model"], device="cpu", compute_type=w["compute_type"], cpu_threads=w["threads"],
                         download_root=str(config.path(w["download_dir"])))
    # decode with our ffmpeg (16 kHz mono) instead of faster-whisper's PyAV: works for any container
    import numpy as np
    r = subprocess.run([ffmpeg_exe(vcfg), "-hide_banner", "-loglevel", "error", "-i", str(media), "-vn",
                        "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True,
                       timeout=vcfg["ffmpeg_timeout_s"])
    if r.returncode != 0 or len(r.stdout) < 16000:  # no audio track (silent clip)
        del model
        return {"source": "none (the video has no audio)", "language": None, "segments": []}
    audio = np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0
    segments, meta = model.transcribe(audio, beam_size=w["beam_size"], vad_filter=True,
                                      language=w["language"] or None)
    out = []
    for s in segments:
        out.append((s.start, s.text.strip()))
        if duration:
            progress(f"transcribing speech {min(99, int(s.end / duration * 100))}%")
    del model
    gc.collect()
    return {"source": f"speech-to-text (Whisper {w['model']})", "language": meta.language, "segments": out}


def get_transcript(src, info, vcfg, folder, progress):
    cache = folder / "transcript.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    tr = None
    if not info.get("local_file"):
        progress("reading captions")
        try:
            tr = get_captions(info, vcfg, folder)
        except Exception as e:
            log("video_captions_failed", error=repr(e))
    if not tr and vcfg["whisper"]["enabled"]:
        progress("downloading audio")
        media = download(src, vcfg, folder, "audio")
        tr = transcribe(media, vcfg, info.get("duration"), progress)
    tr = tr or {"source": "none", "language": None, "segments": []}
    cache.write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
    return tr


def paragraphs(segments, every_s):
    """Caption pieces -> '[mm:ss] text' paragraphs of about every_s seconds."""
    out, cur, start = [], [], None
    for t, s in segments:
        if start is None:
            start = t
        cur.append(s)
        if t - start >= every_s:
            out.append((start, " ".join(cur)))
            cur, start = [], None
    if cur:
        out.append((start, " ".join(cur)))
    return out


# ---------------- 3. visuals ----------------

def pick_frames(cands, n):
    """cands = [(t, path)]. Split the timeline into n slices; in each, keep the frame that differs most
    from the frame before it (a scene change / new slide), skipping black and blank frames."""
    from PIL import Image, ImageStat
    feats = []
    for t, p in cands:
        with Image.open(p) as im:
            g = im.convert("L").resize((24, 24))
            feats.append((t, p, list(g.getdata()), ImageStat.Stat(g).stddev[0]))
    if not feats:
        return []
    dur = feats[-1][0] or 1
    chosen, prev = [], None
    slices = [[] for _ in range(n)]
    for i, (t, p, px, sd) in enumerate(feats):
        change = sum(abs(a - b) for a, b in zip(px, feats[i - 1][2])) / len(px) if i else 255
        if sd > 6:  # not a flat black/white frame
            slices[min(n - 1, int(t / dur * n))].append((change, t, p, px))
    for s in slices:
        if not s:
            continue
        s.sort(key=lambda x: -x[0])
        best = s[0]
        # don't keep a frame that looks the same as the last kept one (static talking head / slide)
        if prev is not None and sum(abs(a - b) for a, b in zip(best[3], prev)) / len(prev) < 4 and len(chosen) > 1:
            continue
        chosen.append((best[1], best[2]))
        prev = best[3]
    return chosen


def extract_frames(media: Path, duration, vcfg, folder, progress):
    fdir = folder / "frames"
    fdir.mkdir(exist_ok=True)
    for old in fdir.glob("c_*.jpg"):
        old.unlink()
    n = max(1, min(vcfg["max_frames"], int((duration or 60) / vcfg["min_seconds_per_frame"])))
    step = max(1.0, (duration or 60) / (n * vcfg["candidates_per_frame"]))
    progress("extracting frames")
    subprocess.run([ffmpeg_exe(vcfg), "-hide_banner", "-loglevel", "error", "-i", str(media),
                    "-vf", f"fps=1/{step:.3f},scale={vcfg['frame_width']}:-2", "-q:v", "3",
                    str(fdir / "c_%05d.jpg")], check=True, timeout=vcfg["ffmpeg_timeout_s"])
    cands = [((i - 1) * step + step / 2, p) for i, p in
             ((int(re.search(r"(\d+)", p.stem).group(1)), p) for p in sorted(fdir.glob("c_*.jpg")))]
    picked = pick_frames(cands, n)
    keep = []
    for t, p in picked:
        dst = fdir / f"frame_{fmt_time(t).replace(':', '-')}.jpg"
        shutil.copy2(p, dst)
        keep.append((t, dst))
    for p in fdir.glob("c_*.jpg"):
        p.unlink()
    return keep


def quiet():
    """The vision library prints its internals (add_text: <|im_start|>..., image_tokens...) straight to
    the console from C code. Silence fd 1/2 while it runs; the spinner/footer use their own console handle."""
    from llama_cpp._utils import suppress_stdout_stderr
    return suppress_stdout_stderr(disable=False)


class Vision:
    """Local vision model, loaded only while frames are described (frees ~3 GB RAM afterwards)."""

    def __init__(self, vcfg, progress):
        v = vcfg["vision"]
        d = config.path(v["models_dir"])
        model, proj = d / v["file"], d / v["mmproj_file"]
        for f in (model, proj):
            if not f.exists():
                progress(f"downloading vision model {f.name}")
                from huggingface_hub import hf_hub_download
                os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
                hf_hub_download(repo_id=v["repo"], filename=f.name, local_dir=d)
        progress("loading vision model")
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        self.v = v
        with quiet():
            self.llm = Llama(model_path=str(model), chat_handler=Qwen25VLChatHandler(clip_model_path=str(proj), verbose=False),
                             n_ctx=v["context_tokens"], n_threads=v["threads"], n_batch=v["batch_size"], verbose=False)

    def describe(self, image: Path, prompt: str) -> str:
        uri = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
        with quiet():
            out = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                                        {"type": "text", "text": prompt}]}],
                max_tokens=self.v["max_tokens"], temperature=self.v["temperature"])
        d = " ".join(out["choices"][0]["message"]["content"].split()).lstrip(":-* ")
        return re.sub(r"^Text:\s*(none|n/?a|-)\.?\s*", "", d, flags=re.I)  # drop an empty 'Text: none'

    def close(self):
        try:
            self.llm.close()
        except Exception:
            pass
        del self.llm
        gc.collect()


def get_visuals(src, info, transcript, vcfg, folder, progress):
    cache = folder / "visuals.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if info.get("is_live"):
        return []
    progress("downloading video (small size, for frames)")
    media = download(src, vcfg, folder, "video")
    duration = info.get("duration") or media_duration(media, vcfg)
    frames = extract_frames(media, duration, vcfg, folder, progress)
    if not vcfg["keep_video"] and not info.get("local_file"):
        media.unlink(missing_ok=True)
    vis = Vision(vcfg, progress)
    out = []
    try:
        for i, (t, p) in enumerate(frames, 1):
            near = " ".join(s for ts, s in transcript["segments"] if abs(ts - t) <= vcfg["frame_speech_window_s"])
            prompt = text(vcfg["prompts"]["frame"]).format(
                title=info.get("title") or "", time=fmt_time(t), speech=near[:400] or "(no speech)")
            progress(f"looking at frame {i}/{len(frames)} ({fmt_time(t)})")
            t0 = time.monotonic()
            desc = vis.describe(p, prompt)
            log("video_frame", time=fmt_time(t), seconds=round(time.monotonic() - t0, 1), description=desc)
            out.append({"t": t, "file": str(p), "description": desc})
            cache.with_suffix(".partial.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    finally:
        vis.close()
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    cache.with_suffix(".partial.json").unlink(missing_ok=True)
    return out


# ---------------- 4. dossier ----------------

def header(info, transcript, visuals, vcfg):
    d = info.get("upload_date") or ""
    date = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
    lines = [f"VIDEO: {info.get('title')}",
             " | ".join(x for x in [
                 f"Channel: {info.get('channel') or info.get('uploader')}" if (info.get("channel") or info.get("uploader")) else "",
                 f"Published: {date}" if date else "",
                 f"Length: {fmt_time(info.get('duration'))}" if info.get("duration") else "",
                 f"Views: {info['view_count']:,}" if info.get("view_count") else "",
                 f"Likes: {info['like_count']:,}" if info.get("like_count") else ""] if x),
             f"Link: {info.get('webpage_url')}",
             f"Speech source: {transcript['source']}" + (f", language: {transcript['language']}" if transcript.get("language") else "")]
    if info.get("chapters"):
        lines.append("Chapters: " + "; ".join(f"[{fmt_time(c['start_time'])}] {c['title']}" for c in info["chapters"]))
    if info.get("tags"):
        lines.append("Tags: " + ", ".join(info["tags"][:15]))
    desc = (info.get("description") or "").strip()
    if desc:
        lines += ["", "DESCRIPTION:", desc[: vcfg["description_chars"]] + ("..." if len(desc) > vcfg["description_chars"] else "")]
    return "\n".join(lines)


def visuals_block(visuals):
    if not visuals:
        return ""
    return "VISUALS (key frames, described by a vision model):\n" + "\n".join(
        f"[{fmt_time(v['t'])}] {v['description']}" for v in visuals)


def comments_block(info, vcfg):
    cs = sorted(info.get("comments") or [], key=lambda c: -(c.get("likes") or 0))[: vcfg["comments_in_dossier"]]
    if not cs:
        return ""
    return "TOP COMMENTS (viewers, not the video):\n" + "\n".join(
        f"- ({c['likes']:,} likes) {' '.join(c['text'].split())[:300]}" for c in cs)


def condense(llm, paras, title, vcfg, budget, count, progress):
    """Transcript too long for the chat model: summarize it chunk by chunk into timestamped notes."""
    chunks, cur, size = [], [], 0
    for t, s in paras:
        line = f"[{fmt_time(t)}] {s}"
        n = count(line)
        if cur and size + n > vcfg["chunk_tokens"]:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += n
    if cur:
        chunks.append(cur)
    per_chunk = max(120, min(vcfg["notes_max_tokens"], budget // max(1, len(chunks))))
    notes = []
    for i, ch in enumerate(chunks, 1):
        progress(f"condensing transcript {i}/{len(chunks)}")
        prompt = text(vcfg["prompts"]["condense"]).format(title=title, part=i, parts=len(chunks), transcript="\n".join(ch))
        out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=per_chunk,
                                         temperature=0.2)["choices"][0]["message"]["content"].strip()
        notes.append(out)
    return "\n".join(notes)


def build_dossier(llm, info, transcript, visuals, vcfg, progress, count):
    paras = paragraphs(transcript["segments"], vcfg["paragraph_seconds"])
    head, vis, com = header(info, transcript, visuals, vcfg), visuals_block(visuals), comments_block(info, vcfg)
    full_tr = "\n".join(f"[{fmt_time(t)}] {s}" for t, s in paras)
    tr_title = "TRANSCRIPT (what is said, with timestamps):"
    full = "\n\n".join(x for x in [head, tr_title + "\n" + (full_tr or "(no speech found)"), vis, com] if x)
    room = vcfg["dossier_tokens"] - count("\n\n".join(x for x in [head, vis, com] if x)) - 50
    if count(full_tr) <= room or not paras:
        return full, full
    notes = condense(llm, paras, info.get("title") or "", vcfg, max(300, room), count, progress)
    short = "\n\n".join(x for x in [head, "TRANSCRIPT NOTES (condensed from the full transcript, with timestamps):\n" + notes, vis, com] if x)
    return short, full


# ---------------- main entry ----------------

def video_id(src) -> str:
    if Path(src).is_file():
        st = Path(src).stat()
        return "file_" + hashlib.sha1(f"{Path(src).resolve()}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()[:12]
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/|embed/)([\w-]{11})", src)
    return ("yt_" + m.group(1)) if m else "url_" + hashlib.sha1(src.encode()).hexdigest()[:12]


def analyze(src, llm, cfg, progress, log_line=print):
    """Returns (dossier for the chat, folder). Every stage is cached, so asking about the same
    video again (or re-running after Ctrl+C) skips what was already done."""
    vcfg = cfg["video"]
    folder = config.path(vcfg["download_dir"]) / video_id(src)
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    count = lambda s: len(llm.tokenize(s.encode("utf-8"), add_bos=False))
    done = folder / "dossier_chat.md"
    if done.exists() and (folder / "info.json").exists():  # analyzed before -> instant
        info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
        log_line(vcfg["texts"]["cached"].format(title=info.get("title")))
        return done.read_text(encoding="utf-8"), folder

    progress("reading video info")
    info = get_info(src, vcfg, folder)
    if vcfg["max_minutes"] and (info.get("duration") or 0) > vcfg["max_minutes"] * 60:
        raise RuntimeError(vcfg["texts"]["too_long"].format(minutes=int(info["duration"] // 60), max=vcfg["max_minutes"]))
    log_line(vcfg["texts"]["found"].format(title=info.get("title"), length=fmt_time(info.get("duration"))))

    transcript = get_transcript(src, info, vcfg, folder, progress)
    log_line(vcfg["texts"]["speech"].format(source=transcript["source"], n=len(transcript["segments"])))

    visuals = []
    if vcfg["vision"]["enabled"]:
        try:
            visuals = get_visuals(src, info, transcript, vcfg, folder, progress)
            log_line(vcfg["texts"]["visuals"].format(n=len(visuals)))
        except KeyboardInterrupt:
            raise
        except Exception as e:  # frames are a bonus: never lose the transcript because of them
            log("video_visuals_failed", error=repr(e))
            log_line(vcfg["texts"]["visuals_failed"].format(error=e))

    short, full = build_dossier(llm, info, transcript, visuals, vcfg, progress, count)
    (folder / "dossier.md").write_text(full, encoding="utf-8")
    (folder / "dossier_chat.md").write_text(short, encoding="utf-8")
    log("video_done", src=src, folder=str(folder), seconds=round(time.monotonic() - t0, 1),
        transcript_source=transcript["source"], segments=len(transcript["segments"]), frames=len(visuals),
        dossier_tokens=count(short), condensed=short != full)
    return short, folder
