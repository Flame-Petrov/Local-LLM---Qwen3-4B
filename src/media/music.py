"""
Music downloader: YouTube / YouTube Music songs and playlists -> tagged audio files (mp3 by default).

    /music <playlist or song link> [check] [force] [single] [max=N] [format=mp3|m4a|opus] [to="D:\\Music"]
           [folders=separate|id|merge]   two playlists with the same name (e.g. two "Mix - DARA" mixes):
           separate = its own folder "Mix - DARA (2)" (default) · id = always "Mix - DARA [<playlist id>]" ·
           merge = one folder, the numbering continues after the last file
    or just paste a playlist link, or "download this playlist as mp3 <link>" / "свали песните от <link>"
    /music rescan                      rebuild the library index from the files on disk
    /music pending | get ... | skip ...  (the skipped songs - see music_followup.py)

Pipeline
  1. resolve    yt-dlp reads the playlist "flat": ids, titles, channels, lengths - nothing is downloaded
  2. library    downloads/music/library.json + an incremental scan of the music folders (only new or
                changed files are read), so moved / renamed / deleted / hand-copied files are noticed
  3. preflight  EVERY song is checked before the first download: the same video already downloaded
                (level 1, by the YouTube id stored in the file), the same song from another video or
                your own file (level 2, artist + title + length), repeated in the playlist (level 3),
                not available, too long. Each warning shows the absolute path of the file you have.
  4. download   one song at a time: full metadata -> late duplicate check -> yt-dlp + ffmpeg ->
                tags (title, artist, album, track, cover, youtube_id) -> library saved after every song
  5. report     on screen (colours, clickable paths) + plain text for the chat history;
                <folder>/playlist.json (or downloads/music/_reports/... in check mode)
Skipped songs stay "open" afterwards (music_followup.py): "download them anyway", "download #15"...
All texts, word lists and settings: config.json -> "music". Network settings (cookies, JS runtime,
timeout) are shared with "video" through video.ydl_opts().
"""

import bisect
import difflib
import glob
import json
import math
import os
import random
import re
import shutil
import time
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from core import config
from core import ui
from media import video
from core.debuglog import log

LIBRARY_VERSION = 1
TAG_ID = "youtube_id"           # tag name of the YouTube video id stored in every downloaded file
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
                  *(f"COM{i}" for i in range(0, 10)), *(f"LPT{i}" for i in range(0, 10))}
_DASH = re.compile(r"\s+[-–—]\s+")
_FEAT_BRACKET = re.compile(r"[(\[{]\s*(?:feat|ft|featuring|with)\b[^)\]}]*[)\]}]")
_FEAT = re.compile(r"(?<!\w)(?:feat|ft|featuring)(?!\w)\.?[^()\[\]{}]*")
_BRACKET = re.compile(r"[(\[{【（]([^()\[\]{}【】（）]*)[)\]}】）]")


class _QuietLogger:
    """yt-dlp writes errors to stderr even when quiet - that would break the live-input footer.
    Its messages go to the debug log instead (the error itself is still raised)."""

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        log("music_ytdlp_warning", message=str(msg)[:300])

    def error(self, msg):
        log("music_ytdlp_error", message=str(msg)[:500])


# ---------------- request ----------------

def parse_request(message: str, mcfg: dict):
    """(url, options) if the message asks for a music download, else None.
    '/music <link> ...' always; plain text needs a YouTube link AND (a playlist / YouTube Music link,
    or music words like 'download ... mp3'). A plain watch?v= link stays with video analysis."""
    msg = message.strip()
    is_cmd = msg.lower().startswith("/music")
    body = msg[len("/music"):].strip() if is_cmd else msg
    m = re.search(mcfg["url_pattern"], body)
    if not m:
        return None
    url = m.group(0).rstrip(".,;:!?)\"'")
    rest = (body[: m.start()] + " " + body[m.end():]).strip()
    if not is_cmd:
        if not re.search(mcfg["youtube_pattern"], url, re.I):
            return None
        if not (re.search(mcfg["auto_pattern"], url, re.I)
                or any(re.search(p, rest, re.I) for p in mcfg["intent_patterns"])):
            return None
    return url, parse_options(rest, mcfg)


def parse_options(rest: str, mcfg: dict) -> dict:
    opts = {"check": False, "force": False, "single": False, "max": None, "format": None, "to": None,
            "folders": None}
    m = re.search(r"(?<!\w)to=(\"[^\"]+\"|'[^']+'|\S+)", rest, re.I)
    if not m:  # plain language: "... to D:\Music" / "... в D:\Music"
        m = re.search(r"(?<!\w)(?:to|into|в)\s+(\"[^\"]+\"|'[^']+'|[A-Za-z]:[\\/]\S*|\\\\\S+)", rest, re.I)
    if m:
        opts["to"] = m.group(1).strip("\"'")
        rest = rest[: m.start()] + " " + rest[m.end():]
    low = _norm(rest)
    for key, words in mcfg["option_words"].items():
        if key in opts and _words_rx(tuple(words)).search(low):
            opts[key] = True
    m = re.search(r"(?<!\w)max=(\d+)", rest, re.I)
    if m:
        opts["max"] = int(m.group(1))
    m = re.search(r"(?<!\w)format=(\w+)", rest, re.I) or re.search(
        r"(?<![\w=])(" + "|".join(map(re.escape, mcfg["formats"])) + r")(?!\w)", rest, re.I)
    if m and m.group(1).lower() in mcfg["formats"]:
        opts["format"] = m.group(1).lower()
    m = re.search(r"(?<!\w)folders?=(\w+)", rest, re.I)
    if m:
        opts["folders"] = m.group(1).lower()  # checked in run() (an unknown value is reported there)
    return opts


# ---------------- names and song matching ----------------

def _norm(s) -> str:
    return unicodedata.normalize("NFKC", str(s or "")).lower().replace("’", "'").replace("‘", "'")


@lru_cache(maxsize=256)
def _words_rx(words: tuple):
    """Regex that finds any of these words / phrases as whole words. 'abc*' = any word starting with abc."""
    parts = []
    for w in sorted({_norm(w).strip() for w in words if w and w.strip()}, key=len, reverse=True):
        prefix = w.endswith("*")
        stem = w[:-1] if prefix else w
        p = re.escape(stem) + (r"\w*" if prefix else "")
        if re.match(r"\w", stem):
            p = r"(?<!\w)" + p
        if not prefix and re.search(r"\w$", stem):
            p += r"(?!\w)"
        parts.append(p)
    return re.compile("|".join(parts) if parts else r"(?!x)x", re.I)


def first_artist(artist: str) -> str:
    """'Avicii, Aloe Blacc' / 'Avicii feat. X' -> 'Avicii' (yt-dlp joins several artists with ', ')."""
    return re.split(r"\s*[,;]\s*|\s+(?:feat|ft|featuring)(?!\w)\.?\s*", artist or "", flags=re.I)[0].strip()


def clean_channel(channel: str) -> str:
    """'Avicii - Topic' / 'AviciiVEVO' / 'Avicii Official' -> 'Avicii'."""
    c = re.sub(r"\s*-\s*topic\s*$", "", channel or "", flags=re.I).strip()
    for _ in range(2):
        c = re.sub(r"\s*(vevo|official)\s*$", "", c, flags=re.I).strip()
    c = re.sub(r"^official\s+", "", c, flags=re.I).strip()
    return c or (channel or "").strip()


def split_artist(title: str, artist: str | None = None, channel: str | None = None):
    """(artist, title) for display and tags. Artist from the metadata, else from 'Artist - Title',
    else from a YouTube Music 'X - Topic' channel, else ''."""
    title = (title or "").strip()
    parts = _DASH.split(title, maxsplit=1)
    if artist:
        if len(parts) == 2 and _norm(first_artist(parts[0])) == _norm(first_artist(artist)):
            return artist.strip(), parts[1].strip()  # the title repeats the artist
        return artist.strip(), title
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()
    if channel and re.search(r"\s-\s*topic\s*$", channel, re.I):
        return clean_channel(channel), title
    return "", title


def clean_title(title: str, dcfg: dict) -> str:
    """Remove noise like '(Official Video)', '[Lyrics]', '| Avicii' - keeps the case, version words
    ('Remix', 'Live' ...) and 'feat.'."""
    noise = _words_rx(tuple(dcfg["noise_words"]))
    versions = _words_rx(tuple(dcfg["version_words"]))
    t = re.split(r"\s[|｜]\s", title or "")[0]

    def drop(m):
        inner = _norm(m.group(1))
        return " " if noise.search(inner) and not versions.search(inner) else m.group(0)

    t = _BRACKET.sub(drop, t)
    # unbracketed noise only at the END ('Levels Official Video'), so 'Video Killed the Radio Star' stays
    t = re.sub(r"(?:[\s\-–—_|,]*(?:" + noise.pattern + r"))+[\s\-–—_|,]*$", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -–—_|,")
    return t or (title or "").strip()


def song_info(title: str, artist: str | None = None, channel: str | None = None, dcfg: dict = None,
              duration=None) -> dict:
    """Normalized identity of a song: {key, artist, title, versions, duration}.
    key = 'artist title' lowercased, noise / feat. / punctuation removed (Cyrillic kept)."""
    a, t = split_artist(title, artist, channel)
    if not a and channel:
        a = clean_channel(channel)
    versions = sorted({_norm(m.group(0)) for m in _words_rx(tuple(dcfg["version_words"])).finditer(_norm(t))})
    tn = _norm(clean_title(t, dcfg))
    tn = _FEAT_BRACKET.sub(" ", tn)
    tn = _FEAT.sub(" ", tn)
    tn = _words_rx(tuple(dcfg["noise_words"])).sub(" ", tn)
    an = _FEAT.sub(" ", _norm(first_artist(a)))

    def collapse(s):
        return re.sub(r"[\W_]+", " ", s).strip()

    an, tn = collapse(an), collapse(tn)
    return {"key": f"{an} {tn}".strip(), "artist": an, "title": tn, "versions": versions,
            "duration": float(duration) if duration else None}


def song_key(title: str, artist: str | None = None, channel: str | None = None, dcfg: dict = None) -> str:
    return song_info(title, artist, channel, dcfg)["key"]


def is_same_song(a: dict, b: dict, dcfg: dict):
    """(same?, score 0-100) for two song_info() dicts. A version word on one side only ('Remix',
    'Live', 'Acoustic'...) means a different song. Lengths must be within duration_tolerance_s."""
    ka, kb = a.get("key"), b.get("key")
    if not ka or not kb or set(a.get("versions") or []) != set(b.get("versions") or []):
        return False, 0
    da, db = a.get("duration"), b.get("duration")
    known = bool(da) and bool(db)
    close = known and abs(float(da) - float(db)) <= dcfg["duration_tolerance_s"]
    if ka == kb:
        return (close or not known), 100
    need = dcfg["title_similarity"] if known else dcfg["title_similarity_no_duration"]
    if known and not close:
        return False, 0
    sm = difflib.SequenceMatcher(None, ka, kb)
    if sm.real_quick_ratio() < need or sm.quick_ratio() < need:
        return False, 0
    ratio = sm.ratio()
    return ratio >= need, round(ratio * 100)


class SongIndex:
    """Songs sorted by length: a lookup compares only songs within the length tolerance (bisect)
    plus exact-key hits, so 500 playlist entries x 5000 library songs stays well under a second."""

    def __init__(self, dcfg: dict):
        self.dcfg = dcfg
        self.refs = []      # [(song_info, ref)]
        self.by_key = {}
        self.timed = []     # sorted [(duration, i)]
        self.untimed = []

    def add(self, info: dict, ref):
        i = len(self.refs)
        self.refs.append((info, ref))
        if info["key"]:
            self.by_key.setdefault(info["key"], []).append(i)
        if info.get("duration"):
            bisect.insort(self.timed, (info["duration"], i))
        else:
            self.untimed.append(i)

    def find(self, info: dict, skip=None):
        """(ref, score) of the best stored song that is the same song as `info`, else None."""
        if not info.get("key") or not self.refs:
            return None
        cand = set(self.by_key.get(info["key"], []))
        d = info.get("duration")
        if d:
            tol = self.dcfg["duration_tolerance_s"]
            lo = bisect.bisect_left(self.timed, (d - tol, -1))
            hi = bisect.bisect_right(self.timed, (d + tol, len(self.refs)))
            cand.update(i for _, i in self.timed[lo:hi])
        else:
            cand.update(i for _, i in self.timed)
        cand.update(self.untimed)
        best = None
        for i in sorted(cand):
            other, ref = self.refs[i]
            if skip and skip(ref):
                continue
            ok, score = is_same_song(info, other, self.dcfg)
            if ok and (best is None or score > best[1]):
                best = (ref, score)
        return best


def safe_filename(name: str, mcfg: dict) -> str:
    """A Windows-safe file name: no <>:"/\\|?* or control characters, no trailing dots/spaces,
    no reserved names (CON, NUL, COM1...), at most max_filename_chars. Unicode (Cyrillic) is kept."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', " ", str(name or ""))
    s = re.sub(r"\s+", " ", s).strip()
    limit = mcfg["max_filename_chars"]
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0] if " " in s[limit // 2: limit] else s[:limit]
    s = s.rstrip(" .")
    if not s:
        s = "untitled"
    head, dot, rest = s.partition(".")
    if head.upper().strip() in RESERVED_NAMES:
        s = head + "_" + dot + rest  # "CON.txt" -> "CON_.txt" (a suffix at the end would still be reserved)
    return s


def _fit_stem(folder, base: str, fmt: str, mcfg: dict) -> str:
    """Shorten `base` so folder + stem + ' (n)'/'.redownload' suffix + extension stays under MAX_PATH."""
    room = 250 - len(str(folder)) - len(fmt) - 8 - 16  # 16: room for ' (nn)' or '.redownload'
    if len(base) > room:
        base = safe_filename(base[:max(room, 20)], mcfg)
    return base


def unique_stem(folder: Path, stem: str, ext: str, taken: set) -> str:
    """`stem`, or 'stem (2)', 'stem (3)'... if a file with that name already exists (a different song)."""
    cand, n = stem, 2
    while (Path(folder) / f"{cand}.{ext}").exists() or cand.lower() in taken:
        cand, n = f"{stem} ({n})", n + 1
    taken.add(cand.lower())
    return cand


# ---------------- playlist folder ----------------

FOLDER_MODES = ("separate", "id", "merge")


def _folder_parts(folder: Path, mcfg: dict):
    """{playlist id: its number offset} of the playlists saved in this folder (its playlist.json),
    None = no readable playlist.json."""
    try:
        data = json.loads((Path(folder) / mcfg["playlist_report_file"]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    parts = {}
    for d in [*(data.get("earlier") or []), data]:
        if isinstance(d, dict) and d.get("playlist_id"):
            parts[d["playlist_id"]] = int(d.get("number_offset") or 0)
    return parts


def _has_audio(folder: Path, mcfg: dict) -> bool:
    exts = tuple(mcfg["audio_extensions"])
    return any(p.is_file() and p.suffix.lower() in exts for p in Path(folder).iterdir())


def _last_number(folder: Path, mcfg: dict) -> int:
    """The highest leading number of the audio files in the folder ('50 - DARA - Nan.mp3' -> 50)."""
    exts, top = tuple(mcfg["audio_extensions"]), 0
    for p in Path(folder).iterdir():
        m = re.match(r"(\d+)\b", p.name)
        if m and p.suffix.lower() in exts:
            top = max(top, int(m.group(1)))
    return top


def playlist_folder(root: Path, pl: dict, mode: str, mcfg: dict):
    """(folder, number offset) for a playlist. YouTube gives many playlists the same title (every
    'Mix - DARA' radio mix), so the title alone is not enough:
      separate  the folder of the same playlist id is reused (a re-run fills it up); another playlist
                with the same title gets 'Title (2)', 'Title (3)'...
      id        always 'Title [playlist id]'
      merge     always 'Title'; a new playlist continues the numbering after the last file there
    """
    if pl["single"]:
        return root / mcfg["singles_folder"], 0
    pid, title = pl["id"] or "", pl["title"] or pl["id"]
    base = safe_filename(title, mcfg)
    if mode == "id" and pid:
        keep = max(10, mcfg["max_filename_chars"] - len(pid) - 3)
        return root / safe_filename(f"{base[:keep].rstrip()} [{pid}]", mcfg), 0
    if mode == "merge":
        folder = root / base
        if not folder.is_dir():
            return folder, 0
        parts = _folder_parts(folder, mcfg) or {}
        if pid in parts:
            return folder, parts[pid]
        return folder, _last_number(folder, mcfg)
    for n in range(1, 1000):  # separate
        # cut the base first so the " (n)" suffix survives safe_filename's length limit
        keep = max(1, mcfg["max_filename_chars"] - len(f" ({n})"))
        folder = root / (base if n == 1 else safe_filename(f"{base[:keep].rstrip()} ({n})", mcfg))
        if not folder.is_dir():
            return folder, 0
        parts = _folder_parts(folder, mcfg)
        if parts is None and not _has_audio(folder, mcfg):
            return folder, 0  # an empty folder / no songs yet: use it
        if parts and pid in parts:
            return folder, parts[pid]
    return root / safe_filename(f"{base} [{pid}]", mcfg), 0


# ---------------- tags ----------------

def _first(v):
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if v is None:
        return None
    if hasattr(v, "text") and not isinstance(v, str):  # ID3 frame
        v = v.text[0] if v.text else None
    if isinstance(v, (bytes, bytearray)):  # MP4 freeform
        v = bytes(v).decode("utf-8", "ignore")
    s = str(v).strip() if v is not None else ""
    return s or None


def read_tags(path) -> dict:
    """{title, artist, album, duration, youtube_id} from an audio file (missing keys left out)."""
    try:
        import mutagen
        f = mutagen.File(str(path))
    except Exception:
        return {}
    if f is None:
        return {}
    out = {}
    length = getattr(getattr(f, "info", None), "length", None)
    if length:
        out["duration"] = round(float(length), 1)
    tags = getattr(f, "tags", None)
    if tags is None:
        return out
    try:
        from mutagen.id3 import ID3
        from mutagen.mp4 import MP4Tags
        if isinstance(tags, ID3):
            g = lambda k: _first(tags.get(k))  # noqa: E731
            out.update(title=g("TIT2"), artist=g("TPE1"), album=g("TALB"), youtube_id=g(f"TXXX:{TAG_ID}"))
        elif isinstance(tags, MP4Tags):
            g = lambda k: _first(tags.get(k))  # noqa: E731
            out.update(title=g("\xa9nam"), artist=g("\xa9ART"), album=g("\xa9alb"),
                       youtube_id=g(f"----:com.apple.iTunes:{TAG_ID}"))
        else:  # Vorbis comments (opus, ogg, flac)
            def g(k):
                try:
                    return _first(tags.get(k))
                except Exception:
                    return None
            out.update(title=g("title"), artist=g("artist"), album=g("album"), youtube_id=g(TAG_ID))
    except Exception as e:
        log("music_tags_read_failed", path=str(path), error=repr(e))
    return {k: v for k, v in out.items() if v}


def write_tags(path, youtube_id=None, url=None, artist=None, title=None, album=None, track=None, total=None) -> bool:
    """Store the YouTube id (+ url, album, track number, clean title/artist) IN the file, so the
    library can always be rebuilt from the disk. Never fails the download: errors are only logged."""
    path = str(path)
    ext = Path(path).suffix.lower()
    try:
        if ext == ".mp3":
            from mutagen.id3 import ID3, TALB, TIT2, TPE1, TRCK, TXXX, WOAS, ID3NoHeaderError
            try:
                tags = ID3(path)
            except ID3NoHeaderError:
                tags = ID3()
            for frame, value in (("TIT2", title), ("TPE1", artist), ("TALB", album)):
                if value:
                    tags.setall(frame, [{"TIT2": TIT2, "TPE1": TPE1, "TALB": TALB}[frame](encoding=3, text=[value])])
            if track:
                tags.setall("TRCK", [TRCK(encoding=3, text=[f"{track}/{total}" if total else str(track)])])
            if youtube_id:
                tags.setall(f"TXXX:{TAG_ID}", [TXXX(encoding=3, desc=TAG_ID, text=[youtube_id])])
            if url:
                tags.setall("WOAS", [WOAS(url=url)])
            tags.save(path, v2_version=3)
        elif ext in (".m4a", ".mp4"):
            from mutagen.mp4 import MP4, MP4FreeForm
            f = MP4(path)
            if f.tags is None:
                f.add_tags()
            t = f.tags
            if title:
                t["\xa9nam"] = [title]
            if artist:
                t["\xa9ART"] = [artist]
            if album:
                t["\xa9alb"] = [album]
            if track:
                t["trkn"] = [(int(track), int(total or 0))]
            if youtube_id:
                t[f"----:com.apple.iTunes:{TAG_ID}"] = [MP4FreeForm(youtube_id.encode("utf-8"))]
            f.save()
        else:
            import mutagen
            f = mutagen.File(path)
            if f is None:
                return False
            if f.tags is None:
                f.add_tags()
            for k, v in (("title", title), ("artist", artist), ("album", album),
                         ("tracknumber", str(track) if track else None), (TAG_ID, youtube_id), ("website", url)):
                if v:
                    f.tags[k] = [v]
            f.save()
        return True
    except Exception as e:
        log("music_tags_failed", path=path, error=repr(e))
        return False


# ---------------- library index ----------------

def music_roots(mcfg: dict, to: str | None = None):
    """(folder this run saves into, every folder scanned for songs you already have)."""
    base = config.path(mcfg["download_dir"])
    root = config.path(str(Path(to).expanduser())) if to else base
    roots = []
    for r in [base, root] + [config.path(d) for d in mcfg["extra_library_dirs"]]:
        if all(os.path.normcase(str(r)) != os.path.normcase(str(x)) for x in roots):
            roots.append(r)
    return root, roots


def index_path(mcfg: dict) -> Path:
    return config.path(mcfg["download_dir"]) / mcfg["library_file"]


def _is_temp(name: str) -> bool:
    low = name.lower()
    stem = low.rsplit(".", 1)[0]
    return low.endswith((".part", ".ytdl")) or stem.endswith((".temp", ".redownload", ".part"))


def _walk(root, exts: tuple, skip_dirs: set):
    """(path, stat) of every audio file under root (os.scandir: fast even for thousands of files)."""
    stack = [str(root)]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if not e.name.startswith(".") and e.name.lower() not in skip_dirs:
                            stack.append(e.path)
                    elif e.name.lower().endswith(exts) and not _is_temp(e.name):
                        yield e.path, e.stat()
                except OSError:
                    continue


def _key(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


class Library:
    """Every audio file in the music folders: {path, youtube_id, title, artist, key, versions, duration,
    size, mtime, source, playlist, downloaded_at}. Saved as library.json (atomically, after every song)."""

    def __init__(self, mcfg: dict, roots, path: Path):
        self.mcfg, self.dcfg = mcfg, mcfg["duplicates"]
        self.roots = [Path(r) for r in roots]
        self.path = Path(path)
        self.items = {}
        self.by_id = {}
        self.songs = SongIndex(self.dcfg)

    def __len__(self):
        return len(self.items)

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != LIBRARY_VERSION:
                raise ValueError(f"version {data.get('version')}")
            self.items = {_key(it["path"]): it for it in data["items"]}
        except FileNotFoundError:
            self.items = {}
        except Exception as e:  # corrupt / old index: rebuilt from the disk by refresh()
            log("music_library_rebuild", file=str(self.path), error=repr(e))
            self.items = {}
        self._reindex()
        return self

    def _under_root(self, k: str) -> bool:
        return any(k.startswith(_key(r).rstrip("\\/") + os.sep) for r in self.roots)

    def refresh(self, full: bool = False) -> dict:
        """Make the index match the disk: new / changed files are read (tags), unchanged ones are kept,
        files that are gone are dropped. full=True re-reads every file."""
        t = time.monotonic()
        exts = tuple(e.lower() for e in self.mcfg["audio_extensions"])
        skip = {self.mcfg["reports_folder"].lower()}
        seen, read = set(), 0
        for root in self.roots:
            for p, st in _walk(root, exts, skip):
                k = _key(p)
                if k in seen:
                    continue
                seen.add(k)
                old = self.items.get(k)
                if old and not full and old.get("size") == st.st_size and old.get("mtime") == st.st_mtime:
                    continue
                self.items[k] = self._from_file(p, st, old)
                read += 1
        dropped = 0
        for k in list(self.items):
            if k in seen:
                continue
            p = self.items[k]["path"]
            if self._under_root(k) or not os.path.isfile(p):
                del self.items[k]
                dropped += 1
            elif full:  # a file outside today's folders (an earlier to= folder): still re-read it
                self.items[k] = self._from_file(p, os.stat(p), self.items[k])
                read += 1
        self._reindex()
        stats = {"items": len(self.items), "files_read": read, "dropped": dropped,
                 "seconds": round(time.monotonic() - t, 2)}
        log("music_library", roots=[str(r) for r in self.roots], **stats)
        return stats

    def _from_file(self, p, st, old=None) -> dict:
        tags = read_tags(p)
        title, artist = tags.get("title"), tags.get("artist")
        if not title:  # no tags: the file name ('07 - Artist - Title', 'Artist - Title', 'Title')
            title = re.sub(r"^\s*\d{1,3}\s*[-._)]\s*", "", Path(p).stem).strip() or Path(p).stem
            if not artist and not _DASH.search(title):
                parent = Path(p).parent
                if all(_key(parent) != _key(r) for r in self.roots):
                    artist = parent.name  # D:\Music\Moby\Porcelain.mp3 -> Moby
        info = song_info(title, artist, None, self.dcfg, tags.get("duration"))
        old = old or {}
        yid = tags.get("youtube_id") or (old.get("youtube_id") if old.get("size") == st.st_size else None)
        return {"path": str(Path(p)), "youtube_id": yid, "title": title, "artist": artist or "",
                "key": info["key"], "versions": info["versions"], "duration": tags.get("duration"),
                "size": st.st_size, "mtime": st.st_mtime, "source": "downloaded" if yid else "scanned",
                "playlist": old.get("playlist") or tags.get("album"), "downloaded_at": old.get("downloaded_at")}

    def _reindex(self):
        self.by_id, self.songs = {}, SongIndex(self.dcfg)
        for k, it in self.items.items():
            self._index_one(k, it)

    def _index_one(self, k, it):
        if it.get("youtube_id"):
            self.by_id.setdefault(it["youtube_id"], []).append(k)
        self.songs.add({"key": it.get("key", ""), "versions": it.get("versions") or [],
                        "duration": it.get("duration")}, k)

    def add(self, path, youtube_id=None, playlist=None) -> dict:
        """A file this program just wrote (download or copy)."""
        k = _key(path)
        it = self._from_file(path, os.stat(path), self.items.get(k))
        it.update(youtube_id=youtube_id or it["youtube_id"], source="downloaded",
                  playlist=playlist or it.get("playlist"), downloaded_at=datetime.now().isoformat(timespec="seconds"))
        self.items[k] = it
        self._index_one(k, it)
        return it

    def by_video(self, youtube_id: str, prefer=None):
        """The library item of this exact YouTube video whose file still exists (one in `prefer` first)."""
        found = [self.items[k] for k in self.by_id.get(youtube_id, ()) if k in self.items]
        found = [it for it in found if os.path.isfile(it["path"])]
        if prefer is not None:
            found.sort(key=lambda it: _key(Path(it["path"]).parent) != _key(prefer))
        return found[0] if found else None

    def same_song(self, info: dict, exclude_id=None):
        """(item, score) of a file that is probably the same song (a different video or your own file)."""
        def skip(k):
            it = self.items.get(k)
            return (it is None or (exclude_id and it.get("youtube_id") == exclude_id)
                    or not os.path.isfile(it["path"]))
        hit = self.songs.find(info, skip=skip)
        return (self.items[hit[0]], hit[1]) if hit else None

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        data = {"version": LIBRARY_VERSION, "updated": datetime.now().isoformat(timespec="seconds"),
                "items": sorted(self.items.values(), key=lambda it: it["path"].lower())}
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


def item_title(it: dict) -> str:
    a, t = it.get("artist") or "", it.get("title") or Path(it["path"]).stem
    return t if not a or _norm(t).startswith(_norm(a)) else f"{a} - {t}"


# ---------------- 1. reading the playlist ----------------

def _entry(e: dict, i: int, mcfg: dict) -> dict:
    vid = e.get("id") or ""
    url = e.get("webpage_url") or e.get("url") or ""
    if not str(url).startswith("http"):
        url = f"https://www.youtube.com/watch?v={vid}"
    title = e.get("title") or vid
    R = mcfg["texts"]["reasons"]
    reason = None
    marker = mcfg["unavailable_titles"].get(_norm(title).strip())
    if marker:
        reason = R[marker]
    elif not e.get("title") and not e.get("duration") and not (e.get("channel") or e.get("uploader")):
        reason = R["hidden"]  # today's flat playlist data for private / deleted videos: no title, length or channel
    elif e.get("live_status") == "is_upcoming":
        reason = R["upcoming"]
    elif e.get("live_status") == "is_live":
        reason = R["live"]
    artists = e.get("artists") or ([e["artist"]] if e.get("artist") else None)
    return {"index": i, "id": vid, "url": url, "title": title,
            "channel": e.get("channel") or e.get("uploader") or "", "duration": e.get("duration"),
            "live_status": e.get("live_status"), "available": reason is None, "reason": reason,
            "artist": ", ".join(artists) if artists else None, "track": e.get("track")}


def resolve(url: str, opts: dict, mcfg: dict, vcfg: dict) -> dict:
    """The playlist (or single song) as {id, title, single, mix, entries}, without downloading anything."""
    import yt_dlp
    mix = bool(re.search(mcfg["mix_pattern"], url))
    limit = opts.get("max") or mcfg["max_items"]
    if mix:
        limit = min(limit, mcfg["mix_max_items"])
    o = video.ydl_opts(vcfg, extract_flat="in_playlist", skip_download=True, playlistend=limit,
                       noplaylist=bool(opts.get("single")), logger=_QuietLogger())
    t = time.monotonic()
    with yt_dlp.YoutubeDL(o) as y:
        info = y.extract_info(url, download=False)
    single = info.get("_type") != "playlist"
    raw = [info] if single else [e for e in (info.get("entries") or [])]
    entries = [_entry(e or {}, i, mcfg) for i, e in enumerate(raw[:limit], 1)]
    pl = {"id": info.get("id") or "", "title": info.get("title") or info.get("id") or "", "url": url,
          "single": single, "mix": mix, "limit": limit, "entries": entries}
    log("music_playlist", id=pl["id"], title=pl["title"], entries=len(entries), single=single, mix=mix,
        unavailable=sum(not e["available"] for e in entries), seconds=round(time.monotonic() - t, 2))
    return pl


def entry_song(e: dict, dcfg: dict) -> dict:
    if e.get("artist") and e.get("track"):
        return song_info(e["track"], e["artist"], e.get("channel"), dcfg, e.get("duration"))
    return song_info(e["title"], None, e.get("channel"), dcfg, e.get("duration"))


def entry_title(e: dict) -> str:
    """How a song is shown: 'Artist - Track' when YouTube gives them, else the video title."""
    if e.get("artist") and e.get("track"):
        return f"{first_artist(e['artist'])} - {e['track']}"
    return e.get("title") or e.get("id") or "?"


# ---------------- 3. preflight ----------------

def preflight(pl: dict, lib: Library, mcfg: dict, opts: dict, folder: Path) -> dict:
    """Sort every entry BEFORE downloading: queue / same_video / same_song / repeats / unavailable / too_long.
    'anyway_*' = duplicates that are downloaded anyway (force, or duplicates.* = download)."""
    dcfg = mcfg["duplicates"]
    force = bool(opts.get("force"))
    act_video = "download" if force and dcfg["same_video"] != "off" else dcfg["same_video"]
    act_song = "download" if force and dcfg["same_song"] != "off" else dcfg["same_song"]
    max_s = (mcfg["max_song_minutes"] or 0) * 60
    pre = {k: [] for k in ("queue", "same_video", "same_song", "repeats", "unavailable", "too_long",
                           "anyway_video", "anyway_song")}
    seen_ids = {}
    here = SongIndex(dcfg)  # this playlist's songs so far (level 3: the same song twice)
    fkey = _key(folder)
    for e in pl["entries"]:
        if not e["available"]:
            pre["unavailable"].append({"entry": e, "reason": e["reason"]})
            continue
        if e["id"] in seen_ids:
            pre["repeats"].append({"entry": e, "first": seen_ids[e["id"]]})
            continue
        seen_ids[e["id"]] = e["index"]
        info = entry_song(e, dcfg)
        job = {"entry": e, "mode": "download", "force": force}
        anyway = False
        if act_video != "off":
            item = lib.by_video(e["id"], prefer=folder)
            if item:
                d = {"entry": e, "path": item["path"], "here": _key(Path(item["path"]).parent) == fkey}
                if act_video == "skip":
                    pre["same_video"].append(d)
                    here.add(info, e)
                    continue
                pre["anyway_video"].append(d)
                anyway, job["force"] = True, True
                if d["here"]:  # the same file name in the same folder: replaced only when the new one is ready
                    job.update(mode="replace", path=item["path"])
        if max_s and (e["duration"] or 0) > max_s:
            pre["too_long"].append(e)
            continue
        rep = here.find(info)
        if rep:
            pre["repeats"].append({"entry": e, "first": rep[0]["index"], "song": True})
            continue
        here.add(info, e)
        if act_song != "off" and not anyway:
            hit = lib.same_song(info, exclude_id=e["id"])
            if hit:
                item, score = hit
                d = {"entry": e, "path": item["path"], "score": score, "other": item_title(item),
                     "other_duration": item.get("duration")}
                if act_song == "skip":
                    pre["same_song"].append(d)
                    continue
                pre["anyway_song"].append(d)
                job["force"] = True
        pre["queue"].append(job)
    log("music_preflight", new=len(pre["queue"]), same_video=len(pre["same_video"]),
        same_song=len(pre["same_song"]), in_playlist=len(pre["repeats"]), too_long=len(pre["too_long"]),
        unavailable=len(pre["unavailable"]), anyway=len(pre["anyway_video"]) + len(pre["anyway_song"]))
    return pre


# ---------------- printing ----------------

class Out:
    """Screen lines above the spinner, coloured by kind; paths on their own line, clickable."""

    def __init__(self, spin, mcfg: dict):
        self.spin, self.mcfg = spin, mcfg
        self.colors = mcfg["colors_256"]

    def line(self, text: str = "", color: str | None = None):
        self.spin.log(ui.paint(text, self.colors.get(color)) if color else text)

    def path(self, path, template_key: str = "dup_path"):
        p = str(path)
        shown = ui.file_link(p) if self.mcfg["clickable_paths"] else p
        self.spin.log(self.mcfg["texts"][template_key].format(path=ui.paint(shown, self.colors["path"])))


def fmt_len(sec) -> str:
    return video.fmt_time(sec) if sec else "?"


def _show_preflight(pl: dict, pre: dict, lib: Library, ctx: dict):
    T, o, cap = ctx["mcfg"]["texts"], ctx["out"], ctx["mcfg"]["duplicates"]["list_max"]
    entries = pl["entries"]
    total_len = sum(e["duration"] or 0 for e in entries)
    if pl["single"]:
        o.line(T["single"].format(title=entry_title(entries[0]) if entries else pl["title"], length=fmt_len(total_len)))
    else:
        o.line(T["playlist"].format(title=pl["title"], n=len(entries), length=fmt_len(total_len)))
    if pl["mix"]:
        o.line(T["mix_note"].format(n=pl["limit"]), "dim")
    dirs = ", ".join(str(r) for r in lib.roots if r.is_dir()) or str(ctx["root"])
    o.line(T["library"].format(n=len(lib), dirs=dirs))

    if pl["single"]:
        for d in pre["same_video"]:
            o.line("")
            o.line(T["single_dup"].format(title=entry_title(d["entry"])), "warn")
            o.path(d["path"])
        for d in pre["same_song"]:
            o.line("")
            o.line(T["single_dup_song"].format(title=entry_title(d["entry"]), other=d["other"], score=d["score"],
                                               len_a=fmt_len(d["entry"]["duration"]), len_b=fmt_len(d["other_duration"])), "warn")
            o.path(d["path"])
    else:
        def group(head, rows, line_fn, color="warn"):
            if not rows:
                return
            o.line("")
            o.line(head, color)
            for d in rows[:cap]:
                line_fn(d)
            if len(rows) > cap:
                o.line(T["dup_more"].format(n=len(rows) - cap, file=ctx["report_path"]), "dim")

        def video_line(d):
            o.line(T["dup_line"].format(index=_num(d["entry"], ctx), title=entry_title(d["entry"])))
            o.path(d["path"])

        def song_line(d):
            o.line(T["dup_song_line"].format(index=_num(d["entry"], ctx), title=entry_title(d["entry"]), other=d["other"],
                                             score=d["score"], len_a=fmt_len(d["entry"]["duration"]),
                                             len_b=fmt_len(d["other_duration"])))
            o.path(d["path"])

        here = [d for d in pre["same_video"] if d["here"]]
        other = [d for d in pre["same_video"] if not d["here"]]
        group(T["dup_here_head"].format(n=len(here)), here, video_line)
        group(T["dup_video_head"].format(n=len(other)), other, video_line)
        group(T["dup_song_head"].format(n=len(pre["same_song"])), pre["same_song"], song_line)
        why = T["why_force"] if ctx["opts"].get("force") else None
        group(T["dup_video_head_download"].format(
            n=len(pre["anyway_video"]), why=why or T["why_setting"].format(key="same_video")), pre["anyway_video"], video_line)
        group(T["dup_song_head_download"].format(
            n=len(pre["anyway_song"]), why=why or T["why_setting"].format(key="same_song")), pre["anyway_song"], song_line)
        firsts = {}
        for d in pre["repeats"]:
            firsts.setdefault(d["first"], []).append(d["entry"]["index"])
        if firsts:
            o.line("")
            by_index = {e["index"]: e for e in entries}
            for first, more in sorted(firsts.items()):
                pos = ", #".join(_pad(i, ctx) for i in [first] + more)
                o.line(T["dup_in_playlist"].format(title=entry_title(by_index[first]), times=1 + len(more),
                                                   positions=pos), "warn")
    if pre["unavailable"] or pre["too_long"]:
        o.line("")
    if pre["unavailable"]:
        det = ", ".join(f"#{_num(d['entry'], ctx)} {d['reason']}" for d in pre["unavailable"][:cap])
        o.line(T["unavailable"].format(n=len(pre["unavailable"]), details=det), "error")
    if pre["too_long"]:
        det = ", ".join(f"#{_num(e, ctx)} {entry_title(e)} ({fmt_len(e['duration'])})" for e in pre["too_long"][:cap])
        o.line(T["too_long"].format(max=ctx["mcfg"]["max_song_minutes"], n=len(pre["too_long"]), details=det), "error")


def _pad(i, ctx) -> str:
    return str(i).zfill(ctx["pad"])


def _num(e, ctx) -> str:
    return _pad(e["index"], ctx)


def _file_num(e, ctx) -> int:
    """The number in the file name / track tag: the playlist position, + the offset of a merged folder."""
    return e["index"] + ctx["offset"]


# ---------------- 4. downloading ----------------

def classify_error(msg: str, mcfg: dict):
    """(kind, reason) for a yt-dlp error. kind: unavailable (can't be fixed) | failed | rate_limit | network."""
    text = str(msg or "")
    R = mcfg["texts"]["reasons"]
    for p in mcfg["error_patterns"]:
        if re.search(p["pattern"], text, re.I):
            return p["kind"], R[p["reason"]]
    first = clean_error(text.strip()).splitlines()
    return "unknown", (first[0][:120] if first and first[0] else R["unknown"])


def clean_error(text: str) -> str:
    """'ERROR: [youtube:tab] LL: YouTube said: The playlist does not exist.' -> 'The playlist does not exist.'"""
    return re.sub(r"^(ERROR:\s*)?(\[[^\]]+\]\s*)?([\w-]+:\s+)?(YouTube said:\s*)?", "", text or "", flags=re.I)


def _artists(info: dict):
    a = info.get("artists") or info.get("creators")
    if isinstance(a, list) and a:
        return ", ".join(str(x) for x in a)
    return info.get("artist") or info.get("creator")


def _song_from_info(info: dict, e: dict, mcfg: dict):
    """(display title, tag artist, tag title, song_info) from the full metadata of one video."""
    dcfg = mcfg["duplicates"]
    channel = info.get("channel") or info.get("uploader") or e.get("channel")
    artist, track = _artists(info), info.get("track")
    if artist and track:
        a, t = first_artist(artist) if "," in artist else artist, track
    else:
        a, t = split_artist(info.get("title") or e["title"], None, channel)
    if mcfg["clean_titles"]:
        t = clean_title(t, dcfg)
    shown = f"{a} - {t}" if a else t
    tag_artist = a or clean_channel(channel or "")
    s = song_info(t, a or None, channel, dcfg, info.get("duration") or e.get("duration"))
    return shown, tag_artist, t, s


def _pp_list(fmt: str, mcfg: dict) -> list:
    pps = [{"key": "FFmpegExtractAudio", "preferredcodec": fmt,
            "preferredquality": str(mcfg["mp3_quality_kbps"]) if fmt == "mp3" else None}]
    if mcfg["embed_thumbnail"]:
        pps.append({"key": "FFmpegThumbnailsConvertor", "format": "jpg", "when": "before_dl"})
    pps.append({"key": "FFmpegMetadata", "add_metadata": True, "add_chapters": False})
    if mcfg["embed_thumbnail"]:
        pps.append({"key": "EmbedThumbnail", "already_have_thumbnail": False})
    return pps


class Job:
    """A running download: the queue can change while it runs (live input: 'get #15 too',
    'don't download #12', 'stop')."""

    def __init__(self, ctx: dict, queue: list, result: dict, known: dict):
        self.ctx, self.queue, self.result = ctx, list(queue), result
        self.known = known      # playlist number -> {"entry", "category", "path", "score"} of skipped songs
        self.pos = 0
        self.stop = False

    def items(self) -> list:
        """Every song this job knows about, as follow-up items (music_followup parses against these)."""
        out = {}
        for n, k in self.known.items():
            out[n] = _pending_item(k["entry"], k["category"], k.get("path"), k.get("score"), k.get("reason"),
                                   k.get("other"), k.get("other_duration"), self.ctx["mcfg"])
        for j in self.queue[self.pos:]:
            e = j["entry"]
            out[e["index"]] = {**_pending_item(e, "queued", None, None, None, None, None, self.ctx["mcfg"]),
                               "actionable": True}
        for s in self.result["saved"]:
            out.setdefault(s["index"], {"n": s["index"], "id": s["id"], "title": s["title"], "category": "done",
                                        "existing_path": s["path"], "actionable": False, "reason": ""})
        return sorted(out.values(), key=lambda it: it["n"])

    def add(self, n: int) -> bool:
        """Download a skipped song too (appended to the end of the queue)."""
        k = self.known.get(n)
        if not k or k["category"] in ("unavailable", "repeat", "same_video_here"):
            return False
        job = {"entry": k["entry"], "mode": "download", "force": True}
        if k["category"] == "same_video" and _can_copy(k.get("path"), self.ctx):
            job.update(mode="copy", path=k["path"])
        self.queue.append(job)
        del self.known[n]
        _drop_from_result(self.result, n)
        return True

    def remove(self, n: int) -> bool:
        """Don't download a queued song after all."""
        for i in range(self.pos, len(self.queue)):
            if self.queue[i]["entry"]["index"] == n:
                e = self.queue.pop(i)["entry"]
                self.result["user_skipped"].append(e)
                return True
        return False


def _drop_from_result(result: dict, n: int):
    for key in ("same_video", "same_song", "failed"):
        result[key] = [d for d in result[key] if d["entry"]["index"] != n]
    result["too_long"] = [e for e in result["too_long"] if e["index"] != n]


def _can_copy(src, ctx) -> bool:
    return bool(src and ctx["mcfg"]["followup"]["copy_instead_of_download"] and os.path.isfile(src)
                and Path(src).suffix.lower() == "." + ctx["fmt"])


def _cleanup(folder: Path, stem: str, before: set):
    """Delete what a failed / stopped download left behind (.part, .ytdl, .webp, .jpg ...) - only files
    that were not there before this download started."""
    for p in Path(folder).glob(glob.escape(stem) + ".*"):
        if p not in before:
            try:
                p.unlink()
            except OSError:
                pass


def _download_one(info: dict, j: dict, i: int, n: int, ctx: dict, titles):
    """Download + convert + tag one song. Returns the final file path. Raises on failure."""
    import yt_dlp
    mcfg, spin, fmt = ctx["mcfg"], ctx["spin"], ctx["fmt"]
    e = j["entry"]
    shown, tag_artist, tag_title, _ = titles
    if j["mode"] == "replace":
        old = Path(j["path"])
        folder, stem = old.parent, old.stem + ".redownload"
    else:
        folder = ctx["folder"]
        tmpl = mcfg["single_filename"] if ctx["single"] else mcfg["playlist_filename"]
        base = safe_filename(tmpl.format(index=_pad(_file_num(e, ctx), ctx), title=shown), mcfg)
        base = _fit_stem(folder, base, fmt, mcfg)  # stay under Windows MAX_PATH
        stem = unique_stem(folder, base, fmt, ctx["taken"])
    folder.mkdir(parents=True, exist_ok=True)
    ctx["current"] = (folder, stem, set(Path(folder).glob(glob.escape(stem) + ".*")))
    final = {}
    short = shown if len(shown) <= 40 else shown[:39] + "…"

    def on_progress(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            pct = f"{min(99, int(d.get('downloaded_bytes', 0) * 100 / total))}%" if total else ""
            spin.status("music_download", n=i, total=n, title=short, pct=pct)

    def on_pp(d):
        if d.get("status") == "started" and d.get("postprocessor") == "ExtractAudio":
            spin.status("music_convert", n=i, total=n, fmt=fmt)

    fsel = mcfg["download_format_for"].get(fmt) or mcfg["download_format"]
    o = video.ydl_opts(ctx["vcfg"], format=fsel, outtmpl=str(folder / (stem.replace("%", "%%") + ".%(ext)s")),
                       writethumbnail=bool(mcfg["embed_thumbnail"]), postprocessors=_pp_list(fmt, mcfg),
                       progress_hooks=[on_progress], postprocessor_hooks=[on_pp],
                       post_hooks=[lambda p: final.__setitem__("path", p)], logger=_QuietLogger())
    spin.status("music_download", n=i, total=n, title=short, pct="")
    with yt_dlp.YoutubeDL(o) as y:
        y.process_ie_result(info, download=True)
    p = Path(final.get("path") or "")
    if not final.get("path") or not p.is_file() or p.stat().st_size == 0:
        raise RuntimeError(mcfg["texts"]["reasons"]["empty_file"])
    if j["mode"] == "replace":
        old = Path(j["path"])
        if p.suffix.lower() == old.suffix.lower():
            target = old
        else:  # another format: don't overwrite an unrelated file with the new extension, drop the old one
            target = old.with_suffix(p.suffix)
            if target.exists():
                target = old.parent / (unique_stem(old.parent, old.stem, p.suffix.lstrip("."), set()) + p.suffix)
        os.replace(p, target)  # the old file is replaced only now, when the new one is complete
        if target != old:
            try:
                old.unlink()
            except OSError as ex:
                log("music_old_remove_failed", path=str(old), error=repr(ex))
        p = target
    album = None if ctx["single"] else ctx["album"]
    write_tags(p, e["id"], e["url"], tag_artist, tag_title, album=album,
               track=None if ctx["single"] else _file_num(e, ctx),
               total=None if ctx["single"] else ctx["size"] + ctx["offset"])
    return p


def _copy_one(j: dict, i: int, n: int, ctx: dict):
    """Same video already downloaded in another folder: copy that file (same video = same audio)."""
    e, src = j["entry"], Path(j["path"])
    ctx["spin"].status("music_copy", n=i, total=n, title=entry_title(e)[:40])
    tmpl = ctx["mcfg"]["single_filename"] if ctx["single"] else ctx["mcfg"]["playlist_filename"]
    shown = clean_title(entry_title(e), ctx["mcfg"]["duplicates"]) if ctx["mcfg"]["clean_titles"] else entry_title(e)
    base = safe_filename(tmpl.format(index=_pad(_file_num(e, ctx), ctx), title=shown), ctx["mcfg"])
    base = _fit_stem(ctx["folder"], base, src.suffix.lstrip("."), ctx["mcfg"])
    ctx["folder"].mkdir(parents=True, exist_ok=True)
    stem = unique_stem(ctx["folder"], base, src.suffix.lstrip(".").lower(), ctx["taken"])
    dest = ctx["folder"] / f"{stem}{src.suffix.lower()}"
    shutil.copy2(src, dest)
    write_tags(dest, e["id"], e["url"], album=None if ctx["single"] else ctx["album"],
               track=None if ctx["single"] else _file_num(e, ctx),
               total=None if ctx["single"] else ctx["size"] + ctx["offset"])
    return dest


def _process(j: dict, i: int, n: int, ctx: dict, job: Job) -> str:
    """One song: 'ok' | 'dup' | 'fail' | 'stop' (rate limit). Ctrl+C is passed on (after cleanup)."""
    import yt_dlp
    mcfg, T, o, lib, result = ctx["mcfg"], ctx["mcfg"]["texts"], ctx["out"], ctx["lib"], ctx["result"]
    e = j["entry"]
    t0 = time.monotonic()
    num = "" if ctx["single"] else f"#{_num(e, ctx)}"

    def saved(p, how, extra_line=None):
        it = lib.add(p, e["id"], None if ctx["single"] else ctx["album"])
        try:
            lib.save()  # a locked/full library.json must not delete the finished song (H8)
        except OSError as ex:
            log("music_library_save_failed", error=repr(ex))
        size = p.stat().st_size
        result["saved"].append({"index": e["index"], "id": e["id"], "title": entry_title(e), "path": str(p),
                                "bytes": size, "duration": it.get("duration") or e["duration"], "how": how})
        log("music_song", index=e["index"], id=e["id"], status=how, seconds=round(time.monotonic() - t0, 1),
            bytes=size, path=str(p))
        if extra_line:
            extra_line()
        return "ok"

    if j["mode"] == "copy":
        if _can_copy(j.get("path"), ctx):
            try:
                p = _copy_one(j, i, n, ctx)
                o.line(T["copied_line"].format(i=i, n=n, num=num, title=entry_title(e)), "ok")
                o.path(j["path"], "from_path")
                return saved(p, "copied")
            except OSError as ex:
                log("music_copy_failed", src=j["path"], error=repr(ex))
        j = {**j, "mode": "download"}  # copy not possible: download it instead

    attempt = 0
    while True:
        ctx["current"] = None
        try:
            ctx["spin"].status("music_info", n=i, total=n, title=entry_title(e)[:40])
            with yt_dlp.YoutubeDL(video.ydl_opts(ctx["vcfg"], logger=_QuietLogger())) as y:
                info = y.extract_info(e["url"], download=False, process=False)
                if info.get("_type") not in (None, "video"):
                    info = y.extract_info(e["url"], download=False)
            titles = _song_from_info(info, e, mcfg)
            if not j.get("force") and mcfg["duplicates"]["same_song"] == "skip":
                hit = lib.same_song(titles[3], exclude_id=e["id"])  # late check with the full metadata
                if hit:
                    item, score = hit
                    result["same_song"].append({"entry": e, "path": item["path"], "score": score, "late": True,
                                                "other": item_title(item), "other_duration": item.get("duration")})
                    o.line(T["dup_late"].format(i=i, n=n, num=num, title=titles[0], other=item_title(item),
                                                score=score), "warn")
                    o.path(item["path"])
                    log("music_song", index=e["index"], id=e["id"], status="same_song_late", path=item["path"])
                    return "dup"
            p = _download_one(info, j, i, n, ctx, titles)
            ctx["current"] = None  # the file is finished: later errors / Ctrl+C must not delete it
            how = "replaced" if j["mode"] == "replace" else "downloaded"
            mb = f"{p.stat().st_size / 1e6:.1f}"
            length = fmt_len(info.get("duration") or e["duration"])
            if how == "replaced":
                o.line(T["replaced_line"].format(i=i, n=n, num=num, title=titles[0]), "ok")
                return saved(p, how, lambda: o.path(p))
            o.line(T["ok_line"].format(i=i, n=n, num=num, title=titles[0], length=length, mb=mb), "ok")
            return saved(p, how)
        except KeyboardInterrupt:
            if ctx.get("current"):
                _cleanup(*ctx["current"])
            raise
        except Exception as ex:
            if ctx.get("current"):
                _cleanup(*ctx["current"])
            msg = str(ex)
            kind, reason = classify_error(msg, mcfg)
            log("music_song", index=e["index"], id=e["id"], status="error", kind=kind, error=msg[:500],
                attempt=attempt, seconds=round(time.monotonic() - t0, 1))
            if kind == "network" and attempt < mcfg["retries_per_song"]:
                attempt += 1
                time.sleep(3)
                continue
            if kind == "rate_limit":
                ctx["rate_hits"] += 1
                if ctx["rate_hits"] == 1:
                    wait = mcfg["rate_limit_wait_s"]
                    o.line(T["rate_wait"].format(seconds=wait), "warn")
                    for left in range(wait, 0, -1):
                        ctx["spin"].status("waiting", seconds=left)
                        time.sleep(1)
                    continue
                result["rate_limited"] = True
                job.queue.insert(job.pos, j)  # not done: stays open as "new"
                return "stop"
            result["failed"].append({"entry": e, "reason": reason, "kind": kind})
            o.line(T["fail_line"].format(i=i, n=n, num=num, title=entry_title(e), reason=reason), "error")
            ctx["last_fail_kind"] = kind
            return "fail"


def _pause(ctx: dict):
    a, b = ctx["mcfg"]["sleep_between_s"]
    end = time.monotonic() + random.uniform(a, b)
    while True:
        left = end - time.monotonic()
        if left <= 0:
            return
        ctx["spin"].status("music_pause", seconds=math.ceil(left))
        time.sleep(min(0.5, left))


def _live_check(job: Job):
    """Between songs: apply what the user typed in the live footer ('get #15 too', 'stop')."""
    try:
        from media import music_followup
        music_followup.live_adjust(job, job.ctx["cfg"], job.ctx["spin"])
    except Exception as ex:  # live input must never break a download
        log("music_followup_live", error=repr(ex))


def download_jobs(job: Job) -> None:
    ctx, result, o, T = job.ctx, job.result, job.ctx["out"], job.ctx["mcfg"]["texts"]
    from media import music_followup
    music_followup.RUNNING = job
    first, running = True, None
    try:
        while True:
            _live_check(job)
            if job.stop or job.pos >= len(job.queue):
                break
            if not first:
                _pause(ctx)
                _live_check(job)
                if job.stop or job.pos >= len(job.queue):
                    break
            first = False
            j = job.queue[job.pos]
            running = job.pos
            job.pos += 1
            ctx["last_fail_kind"] = None
            status = _process(j, job.pos, len(job.queue), ctx, job)
            running = None
            if status == "stop":
                break
            if status in ("ok", "fail") and len(ctx["fail_log"]) < 10:
                # many "not available" / unknown errors among the first songs = yt-dlp is probably outdated
                ctx["fail_log"].append(status == "fail" and ctx["last_fail_kind"] in ("unavailable", "unknown"))
                if len(ctx["fail_log"]) == 10 and sum(ctx["fail_log"]) > 5:
                    result["update_hint"] = True
    except KeyboardInterrupt:
        result["stopped"] = True
        if running is not None:
            job.pos = running  # the song that was running is not done
    finally:
        music_followup.RUNNING = None
        if job.stop:
            result["stopped"] = True
        result["not_done"] += [j["entry"] for j in job.queue[job.pos:]]
        ctx["spin"].stop()


# ---------------- 5. report ----------------

def _pending_item(e, category, path=None, score=None, reason=None, other=None, other_duration=None, mcfg=None):
    actionable = category in ("same_video_here", "same_video", "same_song", "too_long", "failed", "new")
    return {"n": e["index"], "id": e["id"], "url": e["url"], "title": entry_title(e), "channel": e.get("channel"),
            "duration": e.get("duration"), "artist": e.get("artist"), "track": e.get("track"),
            "category": category, "existing_path": path, "score": score, "reason": reason,
            "other": other, "other_duration": other_duration, "actionable": actionable}


def pending_items(result: dict, mcfg: dict) -> list:
    """Everything still open after a run: skipped duplicates, too long, failed, not downloaded yet."""
    items = []
    for d in result["same_video"]:
        items.append(_pending_item(d["entry"], "same_video_here" if d["here"] else "same_video", d["path"], mcfg=mcfg))
    for d in result["same_song"]:
        items.append(_pending_item(d["entry"], "same_song", d["path"], d["score"], None, d["other"],
                                   d["other_duration"], mcfg))
    for e in result["too_long"]:
        items.append(_pending_item(e, "too_long", mcfg=mcfg))
    for d in result["failed"]:
        items.append(_pending_item(d["entry"], "unavailable" if d["kind"] == "unavailable" else "failed",
                                   reason=d["reason"], mcfg=mcfg))
    for d in result["unavailable"]:
        items.append(_pending_item(d["entry"], "unavailable", reason=d["reason"], mcfg=mcfg))
    for d in result["repeats"]:
        items.append(_pending_item(d["entry"], "repeat", reason=f"#{d['first']}", mcfg=mcfg))
    seen = {it["n"] for it in items}
    for e in result["not_done"]:
        if e["index"] not in seen:
            items.append(_pending_item(e, "new", mcfg=mcfg))
    return sorted(items, key=lambda it: it["n"])


def reason_text(it: dict, mcfg: dict) -> str:
    C = mcfg["texts"]["category_reasons"]
    return C[it["category"]].format(score=it.get("score") or 0, len_a=fmt_len(it.get("duration")),
                                    len_b=fmt_len(it.get("other_duration")), reason=it.get("reason") or "",
                                    max=mcfg["max_song_minutes"], length=fmt_len(it.get("duration")),
                                    first=(it.get("reason") or "").lstrip("#"))


def _report_entries(result: dict) -> list:
    rows = {}

    def put(e, status, **kw):
        rows[e["index"]] = {"index": e["index"], "id": e["id"], "title": entry_title(e), "url": e["url"],
                            "status": status, **kw}
    for e in result.get("entries", []):
        put(e, "new")
    for e in result["not_done"]:
        put(e, "not_downloaded")
    for d in result["same_video"]:
        put(d["entry"], "already_downloaded", existing_path=d["path"])
    for d in result["same_song"]:
        put(d["entry"], "same_song", existing_path=d["path"], score=d["score"])
    for e in result["too_long"]:
        put(e, "too_long")
    for d in result["unavailable"]:
        put(d["entry"], "unavailable", reason=d["reason"])
    for d in result["failed"]:
        put(d["entry"], "failed", reason=d["reason"])
    for d in result["repeats"]:
        put(d["entry"], "repeat", reason=f"same as #{d['first']}")
    for e in result["user_skipped"]:
        put(e, "skipped_by_user")
    for s in result["saved"]:
        rows[s["index"]] = {**rows.get(s["index"], {"index": s["index"], "id": s["id"], "title": s["title"]}),
                            "status": s["how"], "path": s["path"]}
    return [rows[k] for k in sorted(rows)]


def write_report(result: dict, path) -> None:
    if not path:
        return
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"source_url": result["url"], "playlist_id": result["playlist_id"],
                "playlist_title": result["playlist_title"], "date": datetime.now().isoformat(timespec="seconds"),
                "folder": result["folder"], "root": result["root"], "format": result["format"],
                "single": result["single"], "size": result["size"], "number_offset": result.get("offset", 0),
                "folders_mode": result.get("folders_mode"),
                "entries": _report_entries(result), "pending": result.get("pending", [])}
        earlier = _earlier_parts(path, result["playlist_id"])
        if earlier:
            data["earlier"] = earlier
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        log("music_report_failed", path=str(path), error=repr(e))


def _earlier_parts(path: Path, playlist_id: str) -> list:
    """folders=merge: the other playlists already saved in this folder's playlist.json are kept."""
    try:
        old = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    parts = [*(old.get("earlier") or []), {k: v for k, v in old.items() if k != "earlier"}]
    return [d for d in parts if isinstance(d, dict) and d.get("playlist_id") and d["playlist_id"] != playlist_id]


def update_report(path, changes: dict, pending: list) -> None:
    """After a follow-up: per playlist number {status, path} + the new open list."""
    try:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = {r["index"]: r for r in data.get("entries", [])}
        for n, ch in changes.items():
            rows.setdefault(n, {"index": n}).update(ch)
        data["entries"] = [rows[k] for k in sorted(rows)]
        data["pending"] = pending
        data["updated"] = datetime.now().isoformat(timespec="seconds")
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    except (OSError, ValueError) as e:
        log("music_report_failed", path=str(path), error=repr(e))


def _new_result(url, opts, fmt) -> dict:
    return {"url": url, "playlist_id": "", "playlist_title": "", "single": False, "folder": None, "root": None,
            "format": fmt, "size": 0, "offset": 0, "folders_mode": None, "entries": [], "saved": [], "same_video": [], "same_song": [], "repeats": [],
            "unavailable": [], "too_long": [], "failed": [], "not_done": [], "user_skipped": [], "anyway": 0,
            "stopped": False, "rate_limited": False, "disk": None, "check_only": bool(opts.get("check")),
            "error": None, "report_path": None, "update_hint": False, "pending": [], "followup": False}


def _context(cfg, spin, result, folder, root, fmt, album, size, single, opts, lib) -> dict:
    mcfg = cfg["music"]
    return {"cfg": cfg, "mcfg": mcfg, "vcfg": cfg["video"], "spin": spin, "out": Out(spin, mcfg), "lib": lib,
            "result": result, "folder": Path(folder), "root": Path(root), "fmt": fmt, "album": album, "size": size,
            "single": single, "opts": opts, "pad": max(2, len(str((size or 1) + result.get("offset", 0)))),
            "offset": result.get("offset", 0), "taken": set(), "current": None,
            "rate_hits": 0, "fail_log": [], "hint_shown": False, "report_path": result["report_path"]}


def _disk_problem(jobs: list, ctx: dict):
    mcfg = ctx["mcfg"]
    kbps = mcfg["mp3_quality_kbps"] if ctx["fmt"] == "mp3" else mcfg["estimate_kbps"]
    need = sum((j["entry"].get("duration") or 240) for j in jobs if j["mode"] != "copy") * kbps / 8 / 1000
    need += sum(os.path.getsize(j["path"]) / 1e6 for j in jobs if j["mode"] == "copy" and os.path.isfile(j["path"]))
    probe = Path(ctx["folder"])
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free / 1e6
    except OSError:
        return None
    if need > free - mcfg["min_free_disk_mb"]:
        return mcfg["texts"]["disk_space"].format(need=math.ceil(need), free=int(free), drive=probe.anchor or probe)
    return None


def _report_file(result: dict, mcfg: dict, root: Path, folder: Path, single: bool, check: bool) -> Path:
    if not check and not single:
        return folder / mcfg["playlist_report_file"]
    name = safe_filename(f"{datetime.now():%Y-%m-%d_%H-%M}_{result['playlist_title'] or 'song'}", mcfg)
    return root / mcfg["reports_folder"] / f"{name}.json"


def run(url: str, opts: dict, cfg: dict, spin) -> dict:
    """The whole job for a link: read -> check the library -> warn (with paths) -> download -> report."""
    mcfg = cfg["music"]
    t0 = time.monotonic()
    fmt = opts.get("format") or mcfg["audio_format"]
    result = _new_result(url, opts, fmt)
    T = mcfg["texts"]
    browser = cfg["video"]["cookies_from_browser"]
    own = re.search(mcfg["own_list_pattern"], url)
    if own and not browser:  # Liked videos / Watch later / Liked music: YouTube shows them only to you
        result["error"] = T["own_list_needs_login"].format(name=T["own_list_names"].get(own.group(1), own.group(1)))
        log("music_done", url=url, error="own list without cookies")
        return result
    try:
        spin.status("music_playlist")
        pl = resolve(url, opts, mcfg, cfg["video"])
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        kind, reason = classify_error(str(ex), mcfg)
        lines = str(ex).strip().splitlines()
        said = clean_error(lines[0] if lines else "")[:160]  # e.g. "The playlist does not exist."
        result["error"] = reason if kind == "unknown" or not said or said.lower() == reason.lower() \
            else f"{reason} ({said})"
        if re.search(mcfg["needs_cookies_pattern"], str(ex) + " " + url, re.I):
            result["error"] += " - " + (T["cookies_used"].format(browser=browser) if browser else T["needs_cookies"])
        log("music_done", url=url, error=str(ex)[:500], kind=kind)
        return result
    if not pl["entries"]:
        result["error"] = T["empty_playlist"]
        return result

    mode = opts.get("folders") or mcfg["same_title_folders"]
    if mode not in FOLDER_MODES:
        result["error"] = T["bad_folders_mode"].format(value=mode, modes="|".join(FOLDER_MODES))
        return result
    root, roots = music_roots(mcfg, opts.get("to"))
    folder, offset = playlist_folder(root, pl, mode, mcfg)
    size = len(pl["entries"])
    result.update(playlist_id=pl["id"], playlist_title=pl["title"], single=pl["single"], folder=str(folder),
                  root=str(root), size=size, entries=pl["entries"], offset=offset, folders_mode=mode)
    log("music_folder", folder=str(folder), mode=mode, offset=offset)
    result["report_path"] = str(_report_file(result, mcfg, root, folder, pl["single"], result["check_only"]))

    spin.status("music_library")
    lib = Library(mcfg, roots, index_path(mcfg)).load()
    lib.refresh()
    ctx = _context(cfg, spin, result, folder, root, fmt, pl["title"], size, pl["single"], opts, lib)
    pre = preflight(pl, lib, mcfg, opts, folder)
    for key in ("same_video", "same_song", "repeats", "unavailable", "too_long"):
        result[key] = list(pre[key])
    result["anyway"] = len(pre["anyway_video"]) + len(pre["anyway_song"])
    spin.stop()
    _show_preflight(pl, pre, lib, ctx)

    queue = pre["queue"]
    if result["check_only"] or not queue:
        result["not_done"] = [j["entry"] for j in queue] if result["check_only"] else []
        if result["check_only"] and queue:
            o = ctx["out"]
            o.line("")
            o.line(T["check_new_head"].format(n=len(queue)))
            for j in queue[: mcfg["duplicates"]["list_max"]]:
                o.line(T["dup_line"].format(index=_num(j["entry"], ctx), title=entry_title(j["entry"])))
            if len(queue) > mcfg["duplicates"]["list_max"]:
                o.line(T["dup_more"].format(n=len(queue) - mcfg["duplicates"]["list_max"], file=result["report_path"]), "dim")
        return _finish(result, mcfg, t0)

    disk = _disk_problem(queue, ctx)
    if disk:
        result["disk"] = disk
        result["not_done"] = [j["entry"] for j in queue]
        return _finish(result, mcfg, t0)

    o = ctx["out"]
    o.line("")
    o.line(T["downloading_one" if len(queue) == 1 else "downloading"].format(n=len(queue), folder=folder))
    known = {}
    for it in pending_items(result, mcfg):
        known[it["n"]] = {"entry": _entry_of(it, pl), "category": it["category"], "path": it["existing_path"],
                          "score": it["score"], "reason": it["reason"], "other": it["other"],
                          "other_duration": it["other_duration"]}
    download_jobs(Job(ctx, queue, result, known))
    return _finish(result, mcfg, t0)


def _entry_of(it: dict, pl: dict) -> dict:
    return next((e for e in pl["entries"] if e["index"] == it["n"]), None)


def _finish(result: dict, mcfg: dict, t0: float) -> dict:
    result["pending"] = pending_items(result, mcfg)
    if not result["check_only"] and not result["single"] and not Path(result["folder"]).is_dir():
        result["report_path"] = str(_report_file(result, mcfg, Path(result["root"]), Path(result["folder"]),
                                                 result["single"], True))  # nothing saved: don't create the folder
    write_report(result, result["report_path"])
    log("music_done", saved=len(result["saved"]), skipped=len(result["same_video"]) + len(result["same_song"]),
        failed=len(result["failed"]), stopped=result["stopped"], rate_limited=result["rate_limited"],
        folder=result["folder"], check_only=result["check_only"], seconds=round(time.monotonic() - t0, 1))
    return result


# ---------------- follow-up downloads (music_followup.execute) ----------------

def run_jobs(meta: dict, jobs: list, cfg: dict, spin) -> dict:
    """Download / copy / replace exactly these songs of an earlier job (no playlist re-read, no duplicate
    checks - the user asked for them). meta = the open decision (music_followup.PENDING)."""
    mcfg = cfg["music"]
    t0 = time.monotonic()
    result = _new_result(meta["source_url"], {}, meta["format"])
    result.update(playlist_id=meta.get("playlist_id", ""), playlist_title=meta["playlist_title"],
                  single=meta["single"], folder=meta["folder"], root=meta["root"], size=meta["size"],
                  offset=meta.get("offset", 0), report_path=meta.get("report_path"), followup=True)
    spin.status("music_library")
    _, roots = music_roots(mcfg, meta["root"])
    lib = Library(mcfg, roots, index_path(mcfg)).load()
    lib.refresh()
    ctx = _context(cfg, spin, result, meta["folder"], meta["root"], meta["format"], meta["playlist_title"],
                   meta["size"], meta["single"], {"force": True}, lib)
    spin.stop()
    disk = _disk_problem(jobs, ctx)
    if disk:
        result["disk"] = disk
        ctx["out"].line(disk, "error")
        return result
    ctx["out"].line(mcfg["texts"]["downloading_asked"].format(n=len(jobs), folder=meta["folder"]))
    download_jobs(Job(ctx, jobs, result, {}))
    log("music_done", followup=True, saved=len(result["saved"]), failed=len(result["failed"]),
        stopped=result["stopped"], seconds=round(time.monotonic() - t0, 1))
    return result


def report(result: dict, mcfg: dict, color: bool = True) -> str:
    """The closing lines on screen."""
    T, c = mcfg["texts"], mcfg["colors_256"]

    def paint(s, key):
        return ui.paint(s, c[key]) if color else s
    if result["error"]:
        return paint(T["failed"].format(error=result["error"]), "error")
    dups = len(result["same_video"]) + len(result["same_song"])
    lines = []
    if result["disk"]:
        lines.append(paint(result["disk"], "error"))
    if result["check_only"]:
        lines.append(T["check_done"].format(new=len(result["not_done"]), dups=dups))
    elif not result["saved"] and not result["failed"] and not result["not_done"] and not result["disk"]:
        lines.append(paint(T["nothing_new"], "ok") if dups or result["repeats"] else
                     T["nothing_to_do"])
    elif not result["disk"]:
        lines.append(paint(T["done"].format(saved=len(result["saved"]), dups=dups, failed=len(result["failed"]),
                                            folder=result["folder"]), "warn" if result["failed"] else "ok"))
    if result["rate_limited"]:
        lines.append(paint(T["rate_limited"].format(saved=len(result["saved"])), "warn"))
    elif result["stopped"]:
        lines.append(paint(T["stopped"].format(saved=len(result["saved"])), "warn"))
    if result["update_hint"]:
        lines.append(paint(T["update_hint"], "dim"))
    return "\n".join(lines)


def history(result: dict, mcfg: dict) -> str:
    """Plain text for the chat history: what happened + the numbered open songs with reasons and paths,
    so the chat model can answer questions and follow-ups."""
    T = mcfg["texts"]
    if result["error"]:
        return T["failed"].format(error=result["error"])
    dups = len(result["same_video"]) + len(result["same_song"])
    title = result["playlist_title"] if not result["single"] else entry_title(result["entries"][0])
    if result["check_only"]:
        lines = [T["history_head_check"].format(title=title, url=result["url"], new=len(result["not_done"]), dups=dups)]
    else:
        skipped = dups + len(result["too_long"])
        head = T["history_head_single"] if result["single"] else T["history_head"]
        lines = [head.format(title=title, url=result["url"], saved=len(result["saved"]), skipped=skipped,
                             failed=len(result["failed"]), folder=result["folder"])]
        for s in result["saved"][: mcfg["history_saved_max"]]:
            lines.append(T["history_saved_line"].format(index=s["index"], title=s["title"], path=s["path"]))
        if len(result["saved"]) > mcfg["history_saved_max"]:
            lines.append(T["history_more"].format(n=len(result["saved"]) - mcfg["history_saved_max"]))
    if result["stopped"] or result["rate_limited"]:
        lines.append(T["history_stopped"])
    items = result.get("pending") or []
    if items:
        lines.append(T["history_list_head"])
        cap = mcfg["followup"]["max_list_in_prompt"]
        for it in items[:cap]:
            lines.append(history_line(it, mcfg))
        if len(items) > cap:
            lines.append(T["history_more"].format(n=len(items) - cap))
    if result.get("report_path"):
        lines.append(T["history_report"].format(file=result["report_path"]))
    return "\n".join(lines)


def history_line(it: dict, mcfg: dict) -> str:
    s = mcfg["texts"]["history_line"].format(index=it["n"], title=it["title"], reason=reason_text(it, mcfg))
    return s + (f" | {it['existing_path']}" if it.get("existing_path") else "")


def rescan(cfg: dict) -> str:
    """Rebuild the library index from the files on disk (every file's tags read again)."""
    mcfg = cfg["music"]
    _, roots = music_roots(mcfg)
    lib = Library(mcfg, roots, index_path(mcfg)).load()
    lib.refresh(full=True)
    lib.save()
    with_id = sum(1 for it in lib.items.values() if it.get("youtube_id"))
    dirs = ", ".join(str(r) for r in roots if Path(r).is_dir()) or str(roots[0])
    return mcfg["texts"]["rescanned"].format(n=len(lib), with_id=with_id, other=len(lib) - with_id, dirs=dirs)
