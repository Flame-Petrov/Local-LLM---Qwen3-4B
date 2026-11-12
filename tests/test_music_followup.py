"""
Offline tests for music_followup.py: the plan's behavior table (rules, LLM stub, clarify, confirm,
execute with the downloader stubbed). Plain asserts - run:

    .venv\\Scripts\\python.exe tests\\test_music_followup.py
"""

import copy
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from core import config  # noqa: E402
from media import music  # noqa: E402
from media import music_followup as mf  # noqa: E402
from test_music import FakeSpin, run_all  # noqa: E402

BASE = config.load()


def cfg(use_llm=False):
    c = copy.deepcopy(BASE)
    c["music"]["followup"]["use_llm"] = use_llm
    return c


def item(n, title, category, path=None, reason=None, duration=200):
    actionable = category in ("same_video_here", "same_video", "same_song", "too_long", "failed", "new")
    return {"n": n, "id": f"id{n}", "url": f"https://www.youtube.com/watch?v=id{n}", "title": title, "channel": "",
            "duration": duration, "artist": None, "track": None, "category": category, "existing_path": path,
            "score": 96 if category == "same_song" else None, "reason": reason, "other": None, "other_duration": None,
            "actionable": actionable}


def open_list(items=None, turns=0, fmt="mp3"):
    items = items if items is not None else [
        item(2, "Daft Punk - One More Time", "same_video", r"E:\m\Party Mix\07 - Daft Punk - One More Time.mp3"),
        item(4, "Some Song", "failed", reason="age-restricted, needs a login"),
        item(9, "Massive Attack - Teardrop", "same_video_here", r"E:\m\Chill Vibes\09 - Massive Attack - Teardrop.mp3"),
        item(11, "[Private video]", "unavailable", reason="private video"),
        item(15, "Avicii - Levels (Official Video)", "same_song", r"D:\Music\Avicii - Levels.mp3"),
        item(17, "Moby - Porcelain", "same_video", r"D:\Music\Moby\Porcelain.m4a"),
    ]
    mf.PENDING = {"source_url": "https://www.youtube.com/playlist?list=PL1", "playlist_id": "PL1",
                  "playlist_title": "Chill Vibes", "folder": r"E:\m\Chill Vibes", "root": r"E:\m", "format": fmt,
                  "single": False, "size": 24, "report_path": None, "check_only": False, "turns_seen": turns,
                  "items": items, "clarify": None, "confirm": None,
                  "done": {1: {"title": "Tycho - Awake", "path": r"E:\m\Chill Vibes\01 - Tycho - Awake.mp3", "id": "id1"}}}
    return mf.PENDING


ALL = [2, 4, 9, 15, 17]


def ask(msg, c=None, turns=0, items=None, llm=None):
    open_list(items, turns)
    return mf.interpret(msg, llm, c or cfg(), FakeSpin())


def acts(d):
    return (d["action"], sorted(d["items"]), sorted(d["skip_items"])) if d else None


def test_download_all():
    for msg in ("download them anyway", "свали ги все пак", "yes, get them", "get them all", "download anyway"):
        assert acts(ask(msg)) == ("download", ALL, []), (msg, ask(msg))
    assert acts(ask("download the skipped songs", turns=9)) == ("download", ALL, [])   # strong words: any time


def test_yes_window():
    assert acts(ask("yes", turns=0)) == ("download", ALL, [])
    assert acts(ask("да", turns=1)) == ("download", ALL, [])
    assert ask("yes", turns=4) is None                     # 'yes' ten messages later is not about the songs
    assert acts(ask("download #15", turns=9)) == ("download", [15], [])   # numbers always count


def test_numbers():
    for msg in ("download 15 and 17", "#15,#17", "15-17", "само 15 и 17", "get 15, 17 please"):
        assert acts(ask(msg)) == ("download", [15, 17], []), (msg, ask(msg))
    assert acts(ask("download 15 at 320 kbps")) == ("download", [15], [])
    assert acts(ask("download #15")) == ("download", [15], [])


def test_titles():
    assert acts(ask("get the avicii one")) == ("download", [15], [])
    assert acts(ask("download the Daft Punk song")) == ("download", [2], [])
    assert acts(ask("all except moby")) == ("download", [2, 4, 9, 15], [])
    assert acts(ask("всички без 17")) == ("download", [2, 4, 9, 15], [])


def test_clauses():
    assert acts(ask("skip 9 but download the rest")) == ("download", [2, 4, 15, 17], [9])
    assert acts(ask("download 15 anyway and skip the rest")) == ("download", [15], [2, 4, 9, 17])
    assert acts(ask("download the similar ones, not the exact duplicates")) == ("download", [15], [])


def test_categories():
    assert acts(ask("download the similar ones")) == ("download", [15], [])
    assert acts(ask("try the failed ones again")) == ("download", [4], [])
    assert acts(ask("download the duplicates")) == ("download", [2, 9, 17], [])


def test_skip():
    for msg in ("no, skip them", "не, остави ги", "no", "leave it", "it's fine", "skip them"):
        assert acts(ask(msg)) == ("skip", ALL, []), (msg, ask(msg))
    assert acts(ask("don't download 17")) == ("skip", [17], [])


def test_not_about_it():
    c = cfg(use_llm=True)

    class NoLLM:
        def create_chat_completion(self, **k):
            raise AssertionError("the model must not be called for this message")
    for msg in ("what's the weather tomorrow?", "tell me a joke", "how are you"):
        assert ask(msg, c, llm=NoLLM()) is None, msg
    # a new link is a new job, not a follow-up
    assert ask("https://www.youtube.com/playlist?list=PL2", c, llm=NoLLM()) is None
    # "I want ..." right after the report but clearly about something else: the rules don't guess
    assert ask("I want to go to the cinema", cfg()) is None


def test_question_and_preference():
    d = ask("where is the daft punk song?")
    assert d["action"] == "question", d
    long_title = [item(1, "Andrew Willis, Skatepark Engineer", "same_video_here", r"E:\m\x.mp3")]
    assert ask("where is the Andrew Willis song?", items=long_title)["action"] == "question"
    assert acts(ask("download andrew willis", items=long_title, turns=9)) == ("download", [1], [])
    assert ask("my friend andrew has a dog called willis", items=long_title, turns=9) is None
    assert ask("why did you skip 15?")["action"] == "question"
    d = ask("from now on don't skip similar songs")
    assert d["action"] == "preference" and d["pref"] == {"same_song": "download"}, d
    d = ask("винаги сваляй и подобните")
    assert d["action"] == "preference" and d["pref"] == {"same_song": "download"}, d
    d = ask("never download duplicates")
    assert d["action"] == "preference" and d["pref"] == {"same_video": "skip"}, d


def test_clarify():
    items = [item(3, "Bonobo - Black Sands", "same_song", r"D:\Music\Black Sands.mp3"),
             item(5, "Bonobo - Kerala", "same_video", r"D:\Music\Kerala.mp3"),
             item(7, "Tycho - Awake", "failed", reason="network problem")]
    c = cfg()
    d = ask("get the bonobo one", c, items=items)
    assert d["action"] == "clarify" and sorted(d["options"]) == [3, 5], d
    buf = io.StringIO()
    with redirect_stdout(buf):
        mf.execute(d, None, c, FakeSpin())
    assert "Which one?" in buf.getvalue() and mf.PENDING["clarify"]["options"] == d["options"]
    second = d["options"][1]
    d2 = mf.interpret("the second", None, c, FakeSpin())
    assert acts(d2) == ("download", [second], []), d2
    # by option number too
    open_list(items)
    mf.PENDING["clarify"] = {"options": [3, 5], "action": "download"}
    assert acts(mf.interpret("2", None, c, FakeSpin())) == ("download", [5], [])


def test_unknown_and_not_actionable():
    d = ask("download 11")
    assert acts(d) == ("download", [11], [])
    c = cfg()
    calls = []
    real = music.run_jobs
    music.run_jobs = lambda *a, **k: calls.append(a) or {}
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            mf.execute(d, None, c, FakeSpin())
        assert "can't be downloaded: not available: private video" in buf.getvalue(), buf.getvalue()
        assert not calls
    finally:
        music.run_jobs = real
    d = ask("download 99")
    assert d["action"] == "unknown" and d["unknown"] == [99], d


def fake_run_jobs(log):
    def run(meta, jobs, cfg_, spin):
        log.append([(j["entry"]["index"], j["mode"]) for j in jobs])
        saved = [{"index": j["entry"]["index"], "id": j["entry"]["id"], "title": j["entry"]["title"],
                  "path": rf"E:\m\Chill Vibes\{j['entry']['index']:02d}.mp3", "how": "copied" if j["mode"] == "copy"
                  else "replaced" if j["mode"] == "replace" else "downloaded"} for j in jobs]
        return {"saved": saved, "failed": [], "disk": None, "rate_limited": False, "stopped": False}
    return run


def test_execute():
    c = cfg()
    calls = []
    real = music.run_jobs
    music.run_jobs = fake_run_jobs(calls)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            d = ask("download them anyway", c)
            hist = mf.execute(d, None, c, FakeSpin())
        jobs = dict(calls[-1])
        # #2 same video in another folder (mp3) -> copied; #17 is .m4a -> downloaded; #9 is already here
        assert jobs == {2: "copy", 4: "download", 15: "download", 17: "download"}, jobs
        assert "#09 Massive Attack - Teardrop is already in this playlist's folder" in buf.getvalue()
        assert mf.PENDING is None  # everything handled -> closed
        assert "✓ #15" in hist, hist

        with redirect_stdout(io.StringIO()):
            d = ask("download 15 anyway and skip the rest", c)
            out = mf.execute(d, None, c, FakeSpin())
        assert dict(calls[-1]) == {15: "download"}
        assert "kept your existing files for #02, #04, #09, #17" in out, out
        assert mf.PENDING is None

        with redirect_stdout(io.StringIO()):
            d = ask("redownload 2, the file is broken", c)
            mf.execute(d, None, c, FakeSpin())
        assert dict(calls[-1]) == {2: "replace"}, calls[-1]

        with redirect_stdout(io.StringIO()):  # a song downloaded fine earlier: only again / redownload re-gets it
            n = len(calls)
            mf.execute(ask("download 1", c), None, c, FakeSpin())
            assert len(calls) == n
            mf.execute(ask("download 1 again", c), None, c, FakeSpin())
        assert dict(calls[-1]) == {1: "replace"}

        with redirect_stdout(io.StringIO()):
            d = ask("skip 9", c)
            out = mf.execute(d, None, c, FakeSpin())
        assert "#09" in out and "Still open: #02, #04, #15, #17" in out, out
        assert [it["n"] for it in mf.PENDING["items"] if it["actionable"]] == [2, 4, 15, 17]

        with redirect_stdout(io.StringIO()):  # preference: session setting only
            mf.execute(ask("from now on don't skip similar songs", c), None, c, FakeSpin())
        assert c["music"]["duplicates"]["same_song"] == "download"
        assert BASE["music"]["duplicates"]["same_song"] == "skip"
    finally:
        music.run_jobs = real


def test_confirm_big():
    c = cfg()
    items = [item(n, f"Artist {n} - Song number {n}", "new") for n in range(1, 26)]
    calls = []
    real = music.run_jobs
    music.run_jobs = fake_run_jobs(calls)
    try:
        with redirect_stdout(io.StringIO()) as buf:
            d = ask("download all", c, items=items)
            mf.execute(d, None, c, FakeSpin())
        assert not calls and "Download 25 songs" in buf.getvalue()
        d2 = mf.interpret("да", None, c, FakeSpin())
        assert d2["source"] == "confirm" and len(d2["items"]) == 25, d2
        with redirect_stdout(io.StringIO()):
            mf.execute(d2, None, c, FakeSpin())
        assert len(calls[-1]) == 25
        # "no" cancels
        with redirect_stdout(io.StringIO()):
            mf.execute(ask("download all", c, items=items), None, c, FakeSpin())
        assert mf.interpret("no", None, c, FakeSpin())["action"] == "cancel"
    finally:
        music.run_jobs = real


class StubLLM:
    def __init__(self, out):
        self.out, self.calls = out, 0

    def create_chat_completion(self, **k):
        self.calls += 1
        return {"choices": [{"message": {"content": self.out}}]}


def test_llm_resolver():
    c = cfg(use_llm=True)
    msg = "I want the one from the festival"  # rules are unsure (an unknown word, no song named)
    llm = StubLLM("ABOUT_SKIPPED: yes\nACTION: download\nSONGS: 15, 99\nSKIP_SONGS: none\nQUESTION:")
    assert acts(ask(msg, c, llm=llm)) == ("download", [15], []) and llm.calls == 1   # 99 is dropped
    assert ask(msg, c, llm=StubLLM("sure! I'll download it for you")) is None        # garbage -> nothing
    assert ask(msg, c, llm=StubLLM("ABOUT_SKIPPED: no\nACTION: none\nSONGS: none")) is None
    d = ask(msg, c, llm=StubLLM("ABOUT_SKIPPED: yes\nACTION: ask\nSONGS: none\nQUESTION: Which festival song?"))
    assert d["action"] == "clarify" and d["ask"] == "Which festival song?", d
    d = ask(msg, c, llm=StubLLM("ABOUT_SKIPPED: yes\nACTION: download\nSONGS: 99"))
    assert d["action"] == "clarify", d  # nothing valid left -> ask, never act
    d = ask(msg, c, llm=StubLLM("ABOUT_SKIPPED: yes\nACTION: download\nSONGS: same_song"))
    assert acts(d) == ("download", [15], [])
    d = ask(msg, c, llm=StubLLM("ABOUT_SKIPPED: yes\nACTION: none\nSONGS: none"))
    assert d["action"] == "question"


def test_is_music_instruction():
    c = cfg()
    open_list()
    assert mf.is_music_instruction("get #15 too", c)
    assert mf.is_music_instruction("don't download #17", c)
    assert not mf.is_music_instruction("tell me a joke", c)
    assert not mf.is_music_instruction("yes", c)      # weak references are never taken mid-run
    mf.PENDING = None
    assert mf.is_music_instruction("get #15 too", c)  # no list at all: a number + an action word
    assert not mf.is_music_instruction("tell me a joke", c)


def test_live_job():
    """While a download runs: 'get #15 too' appends a skipped song, "don't download #3" removes a queued one."""
    c = cfg()
    spin = FakeSpin()
    result = music._new_result("u", {}, "mp3")
    result.update(size=20)
    ctx = music._context(c, spin, result, r"E:\m\X", r"E:\m", "mp3", "X", 20, False, {}, None)
    e = lambda n, t: {"index": n, "id": f"id{n}", "url": "u", "title": t, "channel": "", "duration": 200,  # noqa: E731
                      "available": True, "reason": None, "artist": None, "track": None}
    queue = [{"entry": e(1, "Tycho - Awake"), "mode": "download", "force": False},
             {"entry": e(3, "Bonobo - Kerala"), "mode": "download", "force": False}]
    known = {15: {"entry": e(15, "Avicii - Levels (Official Video)"), "category": "same_song", "path": r"D:\a.mp3"}}
    job = music.Job(ctx, queue, result, known)
    mf.RUNNING = job
    try:
        assert mf.is_music_instruction("get #15 too", c)
        assert mf.is_music_instruction("don't download #3", c)
        assert mf.is_music_instruction("skip the bonobo one", c)
        assert not mf.is_music_instruction("what time is it", c)
    finally:
        mf.RUNNING = None
    d = mf.analyze("get #15 too", mf._job_context(job, c), c)["decision"]
    assert job.add(d["items"][0]) and job.queue[-1]["entry"]["index"] == 15 and job.queue[-1]["force"]
    assert job.remove(3) and [j["entry"]["index"] for j in job.queue] == [1, 15]


def test_command():
    c = cfg()
    calls = []
    real = music.run_jobs
    music.run_jobs = fake_run_jobs(calls)
    try:
        open_list()
        r = mf.command("pending", "", None, c, FakeSpin())
        assert "#15" in r["show"] and "probably the same song" in r["show"], r
        with redirect_stdout(io.StringIO()):
            r = mf.command("get", "similar", None, c, FakeSpin())
        assert dict(calls[-1]) == {15: "download"} and r["history"]
        with redirect_stdout(io.StringIO()):
            r = mf.command("skip", "", None, c, FakeSpin())
        assert mf.PENDING is None and "kept your existing files" in r["history"]
        assert mf.command("get", "1", None, c, FakeSpin())["show"].startswith("There are no skipped songs")
    finally:
        music.run_jobs = real


def test_prompt_extra():
    c = cfg()
    mf.PENDING = None
    assert mf.prompt_extra(c) == ""
    open_list()
    assert "you cannot download files yourself" in mf.prompt_extra(c)
    q = mf.prompt_extra(c, question=True)
    assert r"D:\Music\Avicii - Levels.mp3" in q and "#2 Daft Punk - One More Time" in q, q


def test_numbers_before_full_stop():
    """H9: a number followed by a full stop still counts."""
    assert acts(ask("download 15, 17.")) == ("download", [15, 17], []), ask("download 15, 17.")
    assert acts(ask("download #15.")) == ("download", [15], [])
    assert acts(ask("skip 17.")) is not None


if __name__ == "__main__":
    sys.exit(1 if run_all(globals()) else 0)
