"""
Offline tests for music.py (no network, no model). Plain asserts - run:

    .venv\\Scripts\\python.exe tests\\test_music.py

Test MP3s are generated with bin/ffmpeg.exe in a temp folder.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import config  # noqa: E402
from core import live  # noqa: E402
from media import music  # noqa: E402
from media import video  # noqa: E402

CFG = config.load()
M = CFG["music"]
D = M["duplicates"]


class FakeSpin:
    def __init__(self):
        self.lines = []

    def log(self, s):
        self.lines.append(s)

    def status(self, *a, **k):
        pass

    def start(self, *a, **k):
        pass

    def stop(self):
        pass


def make_mp3(path: Path, seconds: float = 1.0, freq: int = 440):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([video.ffmpeg_exe(CFG["video"]), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"sine=frequency={freq}:duration={seconds}", "-ar", "22050", "-ac", "1", "-b:a", "32k",
                    str(path)], check=True)
    return path


# ---------------- parse_request ----------------

def test_parse_request():
    pl = "https://www.youtube.com/playlist?list=PLabc123"
    r = music.parse_request(f"/music {pl}", M)
    assert r and r[0] == pl and not r[1]["check"] and r[1]["format"] is None, r
    assert music.parse_request(pl, M)[0] == pl                                   # bare playlist link
    assert music.parse_request("https://music.youtube.com/watch?v=abc123", M)    # any YouTube Music link
    w = "https://www.youtube.com/watch?v=abc123&list=PLx"
    assert music.parse_request(f"download this playlist as mp3 {w}", M)[0] == w
    assert music.parse_request("https://www.youtube.com/watch?v=abc123", M) is None   # video analysis keeps it
    assert music.parse_request(f"what is this video about {w}", M) is None
    assert music.parse_request(f"свали песните от {w}", M)[0] == w
    assert music.parse_request("/music", M) is None
    assert music.parse_request("/music no link here", M) is None
    assert music.parse_request("download mp3 https://example.com/song", M) is None  # not YouTube
    r = music.parse_request(f'/music {pl} check force single max=5 format=m4a to="D:\\My Music"', M)
    o = r[1]
    assert o["check"] and o["force"] and o["single"] and o["max"] == 5 and o["format"] == "m4a", o
    assert o["to"] == "D:\\My Music", o
    o = music.parse_request(f"download this playlist as mp3 {pl} to D:\\Music", M)[1]
    assert o["to"] == "D:\\Music" and o["format"] == "mp3", o
    o = music.parse_request(f"/music {pl} format=flac", M)[1]
    assert o["format"] is None, o
    o = music.parse_request(f"провери {pl}", M)[1]
    assert o["check"], o


# ---------------- song matching ----------------

def test_song_key():
    k = lambda *a, **kw: music.song_key(*a, dcfg=D, **kw)  # noqa: E731
    assert k("Avicii - Levels (Official Video)") == k("Avicii - Levels") == "avicii levels"
    assert k("Song (feat. X) [Lyrics]") == k("Song") == "song"
    assert k("Levels", channel="Avicii - Topic") == k("Avicii - Levels")
    assert k("Levels", channel="AviciiOfficialVEVO") == "avicii levels"
    assert k("Avicii - Levels | Best of 2012") == "avicii levels"
    assert k("Levels", artist="Avicii, Aloe Blacc") == "avicii levels"
    assert k("Avicii - Levels", artist="Avicii") == "avicii levels"   # title repeats the artist
    assert k("Тони Стораро - Сто пъти (Официално видео)") == "тони стораро сто пъти"   # Cyrillic kept
    assert k("Avicii - Levels (Skrillex Remix)") == "avicii levels skrillex remix"
    assert music.song_info("Avicii - Levels (Skrillex Remix)", dcfg=D)["versions"] == ["remix"]


def test_clean_title():
    c = lambda t: music.clean_title(t, D)  # noqa: E731
    assert c("Avicii - Levels (Official Video)") == "Avicii - Levels"
    assert c("Levels [Lyrics] | Avicii") == "Levels"
    assert c("Levels (Skrillex Remix) [Official Video]") == "Levels (Skrillex Remix)"
    assert c("The Buggles - Video Killed the Radio Star") == "The Buggles - Video Killed the Radio Star"
    assert c("Song (feat. X)") == "Song (feat. X)"            # feat. stays in file names
    assert c("Levels Official Video") == "Levels"


def test_is_same_song():
    s = lambda t, d=None, **kw: music.song_info(t, dcfg=D, duration=d, **kw)  # noqa: E731
    same = lambda a, b: music.is_same_song(a, b, D)[0]  # noqa: E731
    assert same(s("Avicii - Levels (Official Video)", 199), s("Avicii - Levels", 198))
    assert music.is_same_song(s("Avicii - Levels", 199), s("Avicii - Levels", 198), D)[1] == 100
    assert not same(s("Avicii - Levels", 199), s("Avicii - Levels", 340))
    assert not same(s("Avicii - Levels", 199), s("Avicii - Levels (Skrillex Remix)", 199))
    assert not same(s("Queen - Bohemian Rhapsody (Live at Wembley)", 360), s("Queen - Bohemian Rhapsody", 360))
    # unknown length: exact key -> same, fuzzy -> needs >= 0.97
    assert same(s("Avicii - Levels"), s("Avicii - Levels (Lyrics)"))
    a, b = s("Moby - Porcelaine"), s("Moby - Porcelain")
    ratio = music.is_same_song(a, b, {**D, "title_similarity_no_duration": 0.0})[1]
    assert 90 <= ratio <= 97, ratio   # 0.966 (shown rounded): below the 0.97 needed without lengths
    assert not same(a, b)
    # with length known, 0.90 is enough
    assert same(s("Moby - Porcelaine", 240), s("Moby - Porcelain", 241))
    assert not same(s("Daft Punk - One More Time", 320), s("Daft Punk - Around The World", 320))


def test_song_index():
    idx = music.SongIndex(D)
    for i in range(3000):
        idx.add(music.song_info(f"Artist {i} - Title number {i}", dcfg=D, duration=100 + i % 400), i)
    idx.add(music.song_info("Avicii - Levels", dcfg=D, duration=198), "levels")
    hit = idx.find(music.song_info("Avicii - Levels (Official Video)", dcfg=D, duration=199))
    assert hit and hit[0] == "levels", hit
    assert idx.find(music.song_info("Avicii - Levels", dcfg=D, duration=260)) is None
    assert idx.find(music.song_info("Avicii - Levels", dcfg=D, duration=199), skip=lambda r: r == "levels") is None


def test_safe_filename():
    f = lambda s: music.safe_filename(s, M)  # noqa: E731
    assert f("CON") == "CON_"
    assert f("nul.mp3") == "nul_.mp3"  # M5: "nul.mp3_" would still be reserved
    assert f("a:b?c") == "a b c"
    assert f("name.") == "name"
    assert f("  ") == "untitled"
    assert len(f("x" * 300)) <= M["max_filename_chars"]
    assert len(f(("word " * 80).strip())) <= M["max_filename_chars"]
    assert f("Тони Стораро - Сто пъти") == "Тони Стораро - Сто пъти"
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "01 - Song.mp3").write_bytes(b"x")
        taken = set()
        assert music.unique_stem(Path(td), "01 - Song", "mp3", taken) == "01 - Song (2)"
        assert music.unique_stem(Path(td), "01 - Song", "mp3", taken) == "01 - Song (3)"   # taken in this run
        assert music.unique_stem(Path(td), "02 - Other", "mp3", taken) == "02 - Other"


def test_classify_error():
    k = lambda m: music.classify_error(m, M)  # noqa: E731
    assert k("ERROR: [youtube] abc: Private video. Sign in if you've been granted access")[0] == "unavailable"
    assert k("ERROR: [youtube] abc: Sign in to confirm your age.")[0] == "failed"
    assert k("ERROR: [youtube] abc: Sign in to confirm you're not a bot")[0] == "rate_limit"
    assert k("HTTP Error 429: Too Many Requests")[0] == "rate_limit"
    assert k("ERROR: [youtube] abc: Video unavailable. This video has been removed")[0] == "unavailable"
    assert k("ERROR: Postprocessing: ffmpeg exited with code 1")[0] == "failed"
    assert k("<urlopen error [Errno 11001] getaddrinfo failed>")[0] == "network"
    kind, reason = k("ERROR: [youtube] abcdefghijk: Something strange happened\nmore")
    assert kind == "unknown" and reason == "Something strange happened", reason


# ---------------- tags + library ----------------

def test_tags_and_library():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "music"
        a = make_mp3(root / "Chill" / "01 - Daft Punk - One More Time.mp3", 1.0, 440)
        b = make_mp3(root / "Chill" / "02 - Moby - Porcelain.mp3", 2.0, 550)
        c = make_mp3(root / "Chill" / "03 - Avicii - Levels.mp3", 3.0, 660)
        for p, yid, artist, title in ((a, "id_daft", "Daft Punk", "One More Time"), (b, "id_moby", "Moby", "Porcelain"),
                                      (c, "id_avicii", "Avicii", "Levels")):
            assert music.write_tags(p, yid, f"https://www.youtube.com/watch?v={yid}", artist, title,
                                    album="Chill", track=1, total=3)
        t = music.read_tags(a)
        assert t["youtube_id"] == "id_daft" and t["title"] == "One More Time" and t["artist"] == "Daft Punk", t
        assert t["album"] == "Chill" and 0.5 < t["duration"] < 1.5, t

        lib_file = root / "library.json"
        lib = music.Library(M, [root], lib_file).load()
        st = lib.refresh()
        assert len(lib) == 3 and st["files_read"] == 3, st
        lib.save()
        assert json.loads(lib_file.read_text(encoding="utf-8"))["version"] == music.LIBRARY_VERSION
        # unchanged files are not read again
        lib = music.Library(M, [root], lib_file).load()
        assert lib.refresh()["files_read"] == 0
        assert lib.by_video("id_moby")["path"] == str(b)
        # moved into a subfolder -> same youtube id, new path
        moved = root / "Other" / b.name
        moved.parent.mkdir()
        shutil.move(str(b), str(moved))
        st = lib.refresh()
        assert lib.by_video("id_moby")["path"] == str(moved), lib.by_video("id_moby")
        assert st["dropped"] == 1 and len(lib) == 3
        # deleted -> gone
        os.remove(c)
        lib.refresh()
        assert lib.by_video("id_avicii") is None and len(lib) == 2
        # file without tags: identity from the file name (and the folder as artist)
        make_mp3(root / "Massive Attack - Teardrop.mp3", 1.0, 300)
        make_mp3(root / "Bonobo" / "Kerala.mp3", 1.0, 350)
        lib.refresh()
        info = music.song_info("Massive Attack - Teardrop (Official Video)", dcfg=D, duration=1.0)
        hit = lib.same_song(info)
        assert hit and hit[0]["path"].endswith("Massive Attack - Teardrop.mp3"), hit
        assert lib.same_song(music.song_info("Bonobo - Kerala", dcfg=D, duration=1.0))
        # level 2 never reports the same video (that is level 1)
        assert lib.same_song(music.song_info("Daft Punk - One More Time", dcfg=D, duration=1.0),
                             exclude_id="id_daft") is None
        # corrupt index -> rebuilt, no crash
        lib.save()
        lib_file.write_text("{ not json", encoding="utf-8")
        lib = music.Library(M, [root], lib_file).load()
        assert len(lib) == 0
        lib.refresh()
        assert len(lib) == 4 and lib.by_video("id_daft")
        # temp / partial files are ignored
        (root / "Chill" / "x.temp.mp3").write_bytes(b"x")
        (root / "Chill" / "y.mp3.part").write_bytes(b"x")
        assert lib.refresh()["files_read"] == 0 and len(lib) == 4


def test_m4a_tags():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "song.m4a"
        subprocess.run([video.ffmpeg_exe(CFG["video"]), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", "sine=duration=1", "-c:a", "aac", str(p)], check=True)
        assert music.write_tags(p, "id_m4a", "https://youtu.be/id_m4a", "Artist", "Title", "Album", 2, 9)
        t = music.read_tags(p)
        assert t["youtube_id"] == "id_m4a" and t["title"] == "Title" and t["album"] == "Album", t


# ---------------- preflight / report ----------------

def entry(i, vid, title, dur=200, channel="", available=True, reason=None):
    return {"index": i, "id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "title": title, "channel": channel,
            "duration": dur, "live_status": None, "available": available, "reason": reason, "artist": None, "track": None}


def test_preflight_and_report():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "music"
        here = root / "Chill Vibes"
        p1 = make_mp3(here / "09 - Massive Attack - Teardrop.mp3", 1.0)
        p2 = make_mp3(root / "Party Mix" / "07 - Daft Punk - One More Time.mp3", 1.0)
        p3 = make_mp3(root / "Avicii - Levels.mp3", 1.0)
        music.write_tags(p1, "id_tear", None, "Massive Attack", "Teardrop")
        music.write_tags(p2, "id_daft", None, "Daft Punk", "One More Time")
        lib = music.Library(M, [root], root / "library.json").load()
        lib.refresh()
        pl = {"id": "PL1", "title": "Chill Vibes", "single": False, "mix": False, "limit": 500, "entries": [
            entry(1, "id_new", "Tycho - Awake", 283),
            entry(2, "id_daft", "Daft Punk - One More Time (Official Video)", 320),
            entry(3, "id_bonobo", "Bonobo - Kerala", 240),
            entry(9, "id_tear", "Massive Attack - Teardrop", 330),
            entry(11, "", "[Private video]", None, available=False, reason="private video"),
            entry(12, "id_long", "Some DJ Mix", 3 * 3600),
            entry(15, "id_lev2", "Avicii - Levels (Official Video)", 1),   # level 2: the untagged Avicii file (1 s)
            entry(16, "id_lev3", "Avicii - Levels (Skrillex Remix)", 1),   # a remix is a different song
            entry(21, "id_bonobo", "Bonobo - Kerala", 240),               # repeated video
            entry(22, "id_bon2", "Bonobo - Kerala (Lyrics)", 241),        # repeated song (other video)
        ]}
        pre = music.preflight(pl, lib, M, {}, here)
        q = [j["entry"]["index"] for j in pre["queue"]]
        assert q == [1, 3, 16], q
        assert [(d["entry"]["index"], d["here"]) for d in pre["same_video"]] == [(2, False), (9, True)]
        assert [d["entry"]["index"] for d in pre["same_song"]] == [15] and pre["same_song"][0]["path"] == str(p3)
        assert [(d["entry"]["index"], d["first"]) for d in pre["repeats"]] == [(21, 3), (22, 3)]
        assert [d["entry"]["index"] for d in pre["unavailable"]] == [11]
        assert [e["index"] for e in pre["too_long"]] == [12]
        # force: duplicates are downloaded again; the one in this folder is replaced (not a second copy)
        pre_f = music.preflight(pl, lib, M, {"force": True}, here)
        modes = {j["entry"]["index"]: j["mode"] for j in pre_f["queue"]}
        assert modes[2] == "download" and modes[9] == "replace" and 15 in modes, modes
        # setting same_song=download: downloaded, but still shown
        m2 = {**M, "duplicates": {**D, "same_song": "download"}}
        pre_s = music.preflight(pl, lib, m2, {}, here)
        assert [d["entry"]["index"] for d in pre_s["anyway_song"]] == [15]

        # the screen output: every duplicate with its absolute path, before any download
        spin = FakeSpin()
        result = music._new_result("https://www.youtube.com/playlist?list=PL1", {}, "mp3")
        result.update(playlist_title="Chill Vibes", folder=str(here), root=str(root), size=22, entries=pl["entries"],
                      report_path=str(here / "playlist.json"))
        ctx = music._context(CFG, spin, result, here, root, "mp3", "Chill Vibes", 22, False, {}, lib)
        music._show_preflight(pl, pre, lib, ctx)
        out = "\n".join(spin.lines)
        for p in (p1, p2, p3):
            assert str(p) in out, (p, out)
        assert "Already in this playlist's folder from an earlier run (1)" in out
        assert "Already downloaded (1)" in out and "Probably the same song" in out
        assert '"Bonobo - Kerala" is in the playlist 3 times (#03, #21, #22)' in out, out
        assert "#11 private video" in out and "Longer than 20 min" in out
        assert "\x1b" not in out  # not a terminal: no colour codes

        for key in ("same_video", "same_song", "repeats", "unavailable", "too_long"):
            result[key] = list(pre[key])
        result["saved"] = [{"index": 1, "id": "id_new", "title": "Tycho - Awake", "path": str(here / "01 - Tycho - Awake.mp3"),
                            "bytes": 5_000_000, "duration": 283, "how": "downloaded"}]
        result["failed"] = [{"entry": pl["entries"][2], "reason": "age-restricted, needs a login", "kind": "failed"}]
        result["pending"] = music.pending_items(result, M)
        cats = {it["n"]: (it["category"], it["actionable"]) for it in result["pending"]}
        assert cats[2] == ("same_video", True) and cats[9] == ("same_video_here", True)
        assert cats[15] == ("same_song", True) and cats[12] == ("too_long", True) and cats[3] == ("failed", True)
        assert cats[11] == ("unavailable", False) and cats[21] == ("repeat", False)
        screen = music.report(result, M, color=False)
        assert "Done: 1 downloaded, 3 already in your library, 1 failed" in screen, screen
        hist = music.history(result, M)
        assert hist.startswith('Playlist "Chill Vibes"') and "#2 Daft Punk - One More Time" in hist, hist
        assert f"| {p2}" in hist and "#15 Avicii - Levels (Official Video) | probably the same song" in hist, hist
        assert "#3 Bonobo - Kerala | failed: age-restricted" in hist and "Full report:" in hist
        assert "\x1b" not in hist

        # playlist.json + follow-up update
        rp = here / "playlist.json"
        music.write_report(result, rp)
        data = json.loads(rp.read_text(encoding="utf-8"))
        st = {r["index"]: r["status"] for r in data["entries"]}
        assert st[1] == "downloaded" and st[2] == "already_downloaded" and st[15] == "same_song" and st[11] == "unavailable"
        music.update_report(rp, {9: {"status": "skipped_by_user"}}, [])
        data = json.loads(rp.read_text(encoding="utf-8"))
        assert {r["index"]: r["status"] for r in data["entries"]}[9] == "skipped_by_user" and data["pending"] == []


def test_check_report():
    result = music._new_result("https://www.youtube.com/playlist?list=PL1", {"check": True}, "mp3")
    e1, e2 = entry(1, "a", "A - One"), entry(2, "b", "B - Two")
    result.update(playlist_title="X", folder="F", root="R", size=2, entries=[e1, e2], not_done=[e1])
    result["same_video"] = [{"entry": e2, "path": "R/F/02 - B - Two.mp3", "here": True}]
    result["pending"] = music.pending_items(result, M)
    assert "Check only - nothing downloaded. 1 new, 1 already in your library." in music.report(result, M, False)
    assert [it["category"] for it in result["pending"]] == ["new", "same_video_here"]


def test_stop_and_ctrl_c():
    """Ctrl+C (or 'stop' typed live) keeps finished songs; the rest stays open as 'new' ('continue')."""
    from media import music_followup
    entries = [entry(i, f"id{i}", f"Artist {i} - Song {i}") for i in (1, 2, 3, 4)]
    real_process, real_pause, real_live = music._process, music._pause, music_followup.live_adjust

    def fake_process(j, i, n, ctx, job):
        if j["entry"]["index"] == 3:
            raise KeyboardInterrupt
        ctx["result"]["saved"].append({"index": j["entry"]["index"], "id": j["entry"]["id"], "title": "t",
                                       "path": "p", "bytes": 1, "duration": 1, "how": "downloaded"})
        return "ok"
    music._process, music._pause = fake_process, lambda ctx: None
    try:
        result = music._new_result("u", {}, "mp3")
        result.update(size=4, folder="F", root="R")
        ctx = music._context(CFG, FakeSpin(), result, "F", "R", "mp3", "X", 4, False, {}, None)
        music.download_jobs(music.Job(ctx, [{"entry": e, "mode": "download"} for e in entries], result, {}))
        assert result["stopped"] and [s["index"] for s in result["saved"]] == [1, 2]
        assert [e["index"] for e in result["not_done"]] == [3, 4], result["not_done"]
        assert [it["category"] for it in music.pending_items(result, M)] == ["new", "new"]
        assert "Stopped by you - 2 songs saved" in music.report(result, M, False)

        def stop_after_first(job, cfg, spin):  # the user typed "stop" in the live footer
            if job.pos == 1:
                job.stop = True
        music_followup.live_adjust = stop_after_first
        result = music._new_result("u", {}, "mp3")
        ctx = music._context(CFG, FakeSpin(), result, "F", "R", "mp3", "X", 4, False, {}, None)
        music.download_jobs(music.Job(ctx, [{"entry": e, "mode": "download"} for e in entries[:2]], result, {}))
        assert result["stopped"] and [s["index"] for s in result["saved"]] == [1]
        assert [e["index"] for e in result["not_done"]] == [2]
        assert music_followup.RUNNING is None
    finally:
        music._process, music._pause, music_followup.live_adjust = real_process, real_pause, real_live


def test_own_private_lists():
    """Liked videos / Watch later without cookies: a clear login message, no network call."""
    import copy
    real = music.resolve
    music.resolve = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not read the playlist"))
    try:
        c = copy.deepcopy(CFG)
        c["video"]["cookies_from_browser"] = ""
        for lst, name in (("LL", "Liked videos"), ("WL", "Watch later")):
            r = music.run(f"https://www.youtube.com/playlist?list={lst}", {}, c, FakeSpin())
            assert name in r["error"] and "cookies_from_browser" in r["error"], r["error"]
        # with cookies set, the failure says which browser's login was used
        c["video"]["cookies_from_browser"] = "edge"
        music.resolve = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("ERROR: [youtube:tab] LL: YouTube said: The playlist does not exist."))
        r = music.run("https://www.youtube.com/playlist?list=LL", {}, c, FakeSpin())
        assert r["error"].startswith("not available (The playlist does not exist.)"), r["error"]
        assert "cookies from edge" in r["error"], r["error"]
    finally:
        music.resolve = real
    assert music.clean_error("ERROR: [youtube:tab] LL: YouTube said: The playlist does not exist.") \
        == "The playlist does not exist."


def test_live_take_if():
    li = live.LiveInput(CFG["live_input"], sys.stdout)
    li.queue = [{"text": "get #15 too", "t": 0, "cmd": False}, {"text": "tell me a joke", "t": 0, "cmd": False},
                {"text": "/exit", "t": 0, "cmd": True}]
    got = li.take_if(lambda s: "#" in s)
    assert got == ["get #15 too"], got
    assert [q["text"] for q in li.queue] == ["tell me a joke", "/exit"]
    assert li.consumed_turn == ["get #15 too"]


def test_find_setting():
    import chat
    chat.CFG = config.load()
    node, name = chat.find_setting("music.duplicates.same_song")
    assert name == "same_song" and node is chat.CFG["music"]["duplicates"]
    assert chat.find_setting("music.duplicates.nope") == (None, None)
    assert chat.find_setting("music._about") == (None, None)
    assert chat.find_setting("temperature")[1] == "temperature"


def test_help_sorted():
    import chat
    cmds = [ln.split()[0] for ln in chat.HELP.splitlines() if ln.startswith("  /")]
    assert cmds == sorted(cmds), cmds
    assert "/music" in cmds and "/help" in cmds
    assert chat.sorted_help(chat.HELP) == chat.HELP


def _saved_playlist(folder: Path, pid: str, files: list, offset: int = 0):
    folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        (folder / f).write_bytes(b"x")
    result = music._new_result("u", {}, "mp3")
    result.update(playlist_id=pid, playlist_title="Mix - DARA", folder=str(folder), root=str(folder.parent),
                  size=len(files), offset=offset)
    music.write_report(result, folder / M["playlist_report_file"])


def test_folders_option():
    assert music.parse_options("", M)["folders"] is None
    assert music.parse_options("folders=merge max=5", M)["folders"] == "merge"
    assert music.parse_options("folder=ID", M)["folders"] == "id"
    assert M["same_title_folders"] == "separate"


def test_playlist_folder_modes():
    a = {"id": "RDEMy8CE7_4GS8PTWHAkwfTEnA", "title": "Mix - DARA", "single": False}
    b = {"id": "RDAOy8CE7_4GS8PTWHAkwfTEnA", "title": "Mix - DARA", "single": False}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # nothing there yet: the plain title in every mode
        assert music.playlist_folder(root, a, "separate", M) == (root / "Mix - DARA", 0)
        assert music.playlist_folder(root, a, "merge", M) == (root / "Mix - DARA", 0)
        assert music.playlist_folder(root, a, "id", M) == (root / f"Mix - DARA [{a['id']}]", 0)
        _saved_playlist(root / "Mix - DARA", a["id"], ["01 - x.mp3", "50 - y.mp3"])
        # separate (default): same playlist again -> same folder; another mix with the same name -> (2)
        assert music.playlist_folder(root, a, "separate", M) == (root / "Mix - DARA", 0)
        assert music.playlist_folder(root, b, "separate", M) == (root / "Mix - DARA (2)", 0)
        _saved_playlist(root / "Mix - DARA (2)", b["id"], ["01 - z.mp3"])
        assert music.playlist_folder(root, b, "separate", M) == (root / "Mix - DARA (2)", 0)
        c = {**b, "id": "RDXX"}
        assert music.playlist_folder(root, c, "separate", M) == (root / "Mix - DARA (3)", 0)
        # merge: the second mix continues after #50; re-running it keeps its own offset
        assert music.playlist_folder(root, b, "merge", M) == (root / "Mix - DARA", 50)
        _saved_playlist(root / "Mix - DARA", b["id"], ["51 - z.mp3"], offset=50)
        data = json.loads((root / "Mix - DARA" / M["playlist_report_file"]).read_text(encoding="utf-8"))
        assert data["playlist_id"] == b["id"] and data["number_offset"] == 50
        assert [d["playlist_id"] for d in data["earlier"]] == [a["id"]]  # the first mix is not forgotten
        assert music.playlist_folder(root, b, "merge", M) == (root / "Mix - DARA", 50)
        assert music.playlist_folder(root, a, "merge", M) == (root / "Mix - DARA", 0)
        # a folder with songs but no playlist.json is someone else's: separate does not write into it
        (root / "Other").mkdir()
        (root / "Other" / "song.mp3").write_bytes(b"x")
        assert music.playlist_folder(root, {**a, "title": "Other"}, "separate", M)[0] == root / "Other (2)"
        # singles never change
        assert music.playlist_folder(root, {**a, "single": True}, "id", M) == (root / M["singles_folder"], 0)


def test_file_numbers_with_offset():
    result = music._new_result("u", {}, "mp3")
    result.update(size=12, offset=50)
    ctx = music._context(CFG, FakeSpin(), result, "f", "r", "mp3", "Mix", 12, False, {}, None)
    e = {"index": 3}
    assert music._pad(music._file_num(e, ctx), ctx) == "53"
    assert music._num(e, ctx) == "03"  # shown numbers stay the playlist position (#3 = "download #3")


def test_library_save_failure_keeps_song():
    """H8: a locked/full library.json after a finished download must not delete the mp3."""
    import types
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        e = entry(1, "vidA", "Artist - Song")

        class FakeYDL:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def extract_info(self, url, download=False, process=True):
                return {"_type": "video", "id": "vidA", "title": "Artist - Song", "duration": 200}

        class Lib:
            def same_song(self, *a, **k):
                return None

            def add(self, p, *a, **k):
                return {"duration": 200}

            def save(self):
                raise PermissionError("library.json is locked")

        def fake_download(info, j, i, n, ctx, titles):
            p = folder / "01 - Artist - Song.mp3"
            ctx["current"] = (folder, p.stem, set())
            p.write_bytes(b"x" * 100)
            return p
        real_dl, real_mod = music._download_one, sys.modules.get("yt_dlp")
        music._download_one = fake_download
        sys.modules["yt_dlp"] = types.SimpleNamespace(YoutubeDL=FakeYDL)
        try:
            result = music._new_result("u", {}, "mp3")
            result.update(size=1, folder=td, root=td)
            ctx = music._context(CFG, FakeSpin(), result, td, td, "mp3", "X", 1, False, {}, Lib())
            st = music._process({"entry": e, "mode": "download"}, 1, 1, ctx, None)
            assert st == "ok", st
            assert (folder / "01 - Artist - Song.mp3").is_file()
            assert [s["index"] for s in result["saved"]] == [1]
        finally:
            music._download_one = real_dl
            if real_mod is None:
                sys.modules.pop("yt_dlp", None)
            else:
                sys.modules["yt_dlp"] = real_mod


def test_safe_filename_reserved_and_folder_suffix():
    assert music.safe_filename("CON.txt", M) == "CON_.txt"
    assert music.safe_filename("COM0", M) == "COM0_"
    long = "word " * 100  # M4: a long title keeps its " (n)" suffix, so (2) and (3) differ
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pl = {"single": False, "id": "PLx", "title": long}
        for _ in range(3):  # each folder holds another playlist's songs -> the next one gets a new name
            folder, _off = music.playlist_folder(root, pl, "separate", M)
            assert not folder.exists(), folder
            folder.mkdir()
            (folder / "x.mp3").write_bytes(b"x")
        names = sorted(p.name for p in root.iterdir())
        assert len(names) == 3 and sum(n.endswith((" (2)", " (3)")) for n in names) == 2, names

def run_all(module_globals):
    tests = [(n, f) for n, f in module_globals.items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_all(globals()) else 0)
