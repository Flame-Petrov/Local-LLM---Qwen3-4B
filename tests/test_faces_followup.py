"""
Offline tests for faces_followup.py: the plan's behavior table (§10.1) in English and Bulgarian - rules, the
weak window, clarify / confirm flows, the LLM stub, execute with fake models. Plain asserts - run:

    .venv\\Scripts\\python.exe tests\\test_faces_followup.py
"""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from photos import face_models  # noqa: E402
from photos import face_sync  # noqa: E402
from photos import faces  # noqa: E402
from photos import faces_followup as ff  # noqa: E402
from media import music_followup  # noqa: E402
import test_faces as TF  # noqa: E402
from test_music import run_all  # noqa: E402


def setup(check=False, maybe=False):
    """Two new people (#1, #2) from a scan; optionally one 'not sure' face of #1 (IMG_0450)."""
    env = TF.Env()
    TF.scene(env)
    r = env.run(check=check)
    with redirect_stdout(io.StringIO()):
        ff.open_context(r, env.cfg)
    if maybe:
        lib = env.open()
        try:
            P1, P2, M = lib.people()[1], lib.people()[2], lib.mcfg
        finally:
            lib.close()
        env.photo("IMG_0450.jpg", [{"emb": TF.borderline(P1, M, 77)}])
        env.photo("IMG_0451.jpg", [{"emb": TF.borderline(P2, M, 78)}])
        env.photo("IMG_0452.jpg", [{"emb": TF.borderline(P1, M, 79)}])
        r = env.run()
        assert len(r["maybe"]) == 3, r["maybe"]
        ff.open_context(r, env.cfg)
    return env


def ask(env, msg, llm=None, fresh=True):
    """fresh: as if right after the report (inside the weak window) - unless the test is about the window."""
    if fresh and ff.CONTEXT and ff.CONTEXT["turns_seen"] < 10 ** 5:
        ff.CONTEXT["turns_seen"] = 0
    return ff.interpret(msg, llm, env.cfg, env.spin)


def run(env, msg, llm=None):
    d = ask(env, msg, llm)
    assert d is not None, msg
    with redirect_stdout(io.StringIO()) as buf:
        out = ff.execute(d, env.cfg, env.spin)
    return d, out, buf.getvalue()


def names(env):
    lib = env.open()
    try:
        return {p["id"]: (p["name"], p["status"]) for p in lib.store.persons()}
    finally:
        lib.close()


def act(env, msg):
    d = ask(env, msg)
    return d["action"] if d else None


# ---------------- rename ----------------

def test_rename_forms():
    env = setup()
    try:
        d, out, _ = run(env, "1 is Maria")
        assert d["action"] == "rename" and d["person"] == 1 and d["name"] == "Maria", d
        assert "Renamed #1 Person 1 → Maria" in out and names(env)[1][0] == "Maria"
        assert "Maria" in env.person_dirs()
        ff.expire_weak()
        assert act(env, "2 is Ivan") is None                          # bare number outside the weak window
        ff.CONTEXT["turns_seen"] = 0
        for msg in ("person 2 is Ivan", "#2 = Ivan", "name 2 Ivan", "човек 2 е Иван", "кръсти 2 Иван", "number 2 is Ivan"):
            d = ask(env, msg)
            assert d and d["action"] == "rename" and d["person"] == 2 and d["name"] in ("Ivan", "Иван"), (msg, d)
        d, out, _ = run(env, "rename Maria to Maria Petrova")
        assert d["action"] == "rename" and d["name"] == "Maria Petrova"
        pj = json.loads((env.people / "Maria Petrova" / "person.json").read_text(encoding="utf-8"))
        assert pj["aliases"] == ["Maria"]
        d = ask(env, "преименувай Maria Petrova на Мария Петрова")
        assert d and d["action"] == "rename" and d["name"] == "Мария Петрова", d
        # lowercase name is capitalized
        d = ask(env, "човек 2 е мария")
        assert d and d["action"] == "rename" and d["name"] == "Мария", d
    finally:
        env.close()


def test_rename_existing_name_asks_merge():
    env = setup()
    try:
        run(env, "1 is Maria")
        d, out, _ = run(env, "person 2 is Maria")
        assert d["action"] == "ask_merge" and "Are they the same person?" in out, (d, out)
        d, out, _ = run(env, "no")
        assert d["action"] == "rename" and names(env)[2][0] == "Maria 2", (d, names(env))
        run(env, "person 2 is Ivan")
        d, out, _ = run(env, "person 2 is Maria")
        assert d["action"] == "ask_merge"
        d, out, _ = run(env, "yes")
        assert d["action"] == "merge" and names(env)[2][1] == "merged", names(env)
    finally:
        env.close()


def test_which_one():
    env = setup()
    try:
        # two people called Maria (e.g. from person.json edits) -> "Which one?"
        lib = env.open()
        try:
            for pid in (1, 2):
                lib.store.update_person(pid, name="Maria", last_name="Maria")
            lib.store.commit()
        finally:
            lib.close()
        d, out, _ = run(env, "rename Maria to Elena")
        assert d["action"] == "ask_which" and "Which one?" in out and "#1" in out and "#2" in out, out
        d = ask(env, "2")
        assert d["action"] == "rename" and d["person"] == 2 and d["name"] == "Elena", d
        # with the number it is clear
        d = ask(env, "ignore #1")
        assert d["action"] == "ignore" and d["person"] == 1
    finally:
        env.close()


# ---------------- merge ----------------

def test_merge_forms():
    env = setup()
    try:
        for msg in ("1 and 2 are the same person", "merge 1 into 2", "1 is also 2", "1 и 2 са един човек"):
            d = ask(env, msg)
            assert d and d["action"] == "merge" and (d["a"], d["b"]) == (1, 2), (msg, d)
        run(env, "1 is Maria")
        d = ask(env, "person 2 is also Maria")
        assert d and d["action"] == "merge" and (d["a"], d["b"]) == (2, 1), d
        # both big -> asks first
        env.cfg["faces"]["followup"]["confirm_merge_over_photos"] = 2
        d, out, _ = run(env, "merge 2 into Maria")
        assert "yes / no" in out and names(env)[2][1] == "active"
        d, out, _ = run(env, "yes")
        assert d["action"] == "merge" and names(env)[2][1] == "merged"
        assert "Merged #2" in out
    finally:
        env.close()


# ---------------- the 'not sure' list ----------------

def test_maybe_answers():
    env = setup(maybe=True)
    try:
        d = ask(env, "1 and 3 yes, 2 no")
        assert d["action"] == "answer" and sorted(d["ops"]) == [(1, "yes", None), (2, "no", None), (3, "yes", None)], d
        d = ask(env, "1 да, 2 не")
        assert d["action"] == "answer" and sorted(d["ops"]) == [(1, "yes", None), (2, "no", None)], d
        d = ask(env, "all yes")
        assert d["action"] == "answer" and len(d["ops"]) == 3
        d = ask(env, "да, всички")
        assert d and d["action"] == "answer" and len(d["ops"]) == 3, d
        assert ask(env, "yes") is None or ask(env, "yes")["action"] != "answer"   # 3 open: which?
        d, out, _ = run(env, "1 yes, 2 is 1")
        assert d["action"] == "answer", d
        assert "✓ IMG_0450.jpg: Person 1" in out and "✓ IMG_0451.jpg: Person 1" in out, out
        assert (env.people / "Person 1" / "IMG_0450.jpg").exists() and (env.people / "Person 1" / "IMG_0451.jpg").exists()
        # one left: a plain "yes" means it
        d, out, _ = run(env, "yes")
        assert d["action"] == "answer" and d["ops"] == [(3, "yes", None)], d
        assert (env.people / "Person 1" / "IMG_0452.jpg").exists()
    finally:
        env.close()


def test_check_does_not_answer_old_list():
    """H7: scan A ('not sure' x3) -> a check report -> '1 yes' must not confirm A's item 1."""
    env = setup(maybe=True)
    try:
        assert len(faces.OPEN_MAYBE) == 3
        r = env.run(check=True)
        with redirect_stdout(io.StringIO()):
            ff.open_context(r, env.cfg)
        assert ff.CONTEXT["maybe"] == [] and ff.CONTEXT["check"]
        d = ask(env, "1 yes")
        assert d is None or d["action"] != "answer", d
        fid = faces.OPEN_MAYBE[0]["face_id"]
        with redirect_stdout(io.StringIO()):
            ff.execute({"action": "answer", "ops": [(1, "yes", None)], "source": "llm"}, env.cfg, env.spin)
        lib = env.open()
        try:
            assert lib.store.face(fid)["status"] == "maybe"
        finally:
            lib.close()
    finally:
        env.close()


def test_rename_with_yes_word():
    """G1: 'ok 3 is Maria' is a rename, not an answer to the 'not sure' list."""
    env = setup(maybe=True)
    try:
        d = ask(env, "ok 2 is Maria")
        assert d is None or d["action"] != "answer", d     # rename or 'unsure' (-> the chat model), never an answer
    finally:
        env.close()


def test_maybe_assign_by_name_and_person_or_item():
    env = setup(maybe=True)
    try:
        run(env, "person 2 is Ivan")
        d, out, _ = run(env, "1 is Ivan")
        assert d["action"] == "answer" and d["ops"] == [(1, "assign", 2)], d
        assert (env.people / "Ivan" / "IMG_0450.jpg").exists()
        # "2 is Ivan" when #2 is a new person AND 'not sure' item 2 exists -> which one?
        env2 = setup(maybe=True)
        try:
            run(env2, "person 1 is Ivan")
            ff.CONTEXT["new_people"] = [1, 2]
            d, out, _ = run(env2, "2 is Ivan")
            assert d["action"] == "ask_person_or_item" and "person #2 or 'not sure' photo 2" in out, (d, out)
            d = ask(env2, "the photo")
            assert d["action"] == "answer" and d["ops"] == [(2, "assign", 1)], d
        finally:
            env2.close()
    finally:
        env.close()


# ---------------- photos ----------------

def test_reject_and_position():
    env = setup()
    try:
        run(env, "1 is Maria")
        for msg in ("IMG_0002 is not Maria", "remove IMG_0002 from Maria", "IMG_0002 не е Maria", "IMG_0002.jpg isn't Maria"):
            d = ask(env, msg)
            assert d and d["action"] == "reject_photo" and d["person"] == 1, (msg, d)
        d, out, _ = run(env, "IMG_0002 is not Maria")
        assert "✗ IMG_0002.jpg: not Maria" in out, out
        assert not (env.people / "Maria" / "IMG_0002.jpg").exists()
        d = ask(env, "in IMG_0005 Maria is the left one")
        assert d and d["action"] == "confirm_position" and d["position"] == "left", d
        d = ask(env, "на IMG_0005 Maria е отляво")
        assert d and d["action"] == "confirm_position", d
        d, out, _ = run(env, "who is in IMG_0005?")
        assert d["action"] == "who_photo" and "Maria" in out and "Person 2" in out, out
    finally:
        env.close()


# ---------------- ignore / forget / sort / thresholds ----------------

def test_ignore_forget():
    env = setup()
    try:
        for msg in ("2 is a stranger", "ignore 2", "игнорирай 2", "не го познавам 2"):
            d = ask(env, msg)
            assert d and d["action"] == "ignore" and d["person"] == 2, (msg, d)
        run(env, "ignore 2")
        assert names(env)[2][1] == "ignored" and "Person 2" not in env.person_dirs()
        d, out, _ = run(env, "forget 1")
        assert d["action"] == "ask_forget" and "yes / no" in out
        d, out, _ = run(env, "no")
        assert d["action"] == "cancel" and 1 in names(env)
        run(env, "forget 1")
        d, out, _ = run(env, "yes")
        assert d["action"] == "forget" and 1 not in names(env), names(env)
        d = ask(env, "delete everything about 2")
        assert d and d["action"] == "ask_forget"
    finally:
        env.close()


def test_sort_after_check():
    env = setup(check=True)
    try:
        assert not env.person_dirs()
        d, out, txt = run(env, "ok, sort them")
        assert d["action"] == "sort" and env.person_dirs() == ["Person 1", "Person 2"], (d, env.person_dirs())
        assert "New people" in txt
        d = ask(env, "давай")
        assert d is None or d["action"] != "sort"      # not a check context any more
    finally:
        env.close()


def test_thresholds_and_preference():
    env = setup()
    try:
        run(env, "1 is Maria")
        d, out, _ = run(env, "be stricter with Maria")
        assert d["action"] == "stricter" and d["person"] == 1 and "0.43" in out, out
        pj = json.loads((env.people / "Maria" / "person.json").read_text(encoding="utf-8"))
        assert pj["recognition"]["match_threshold"] == 0.43
        d, out, _ = run(env, "по-строго")
        assert d["action"] == "stricter" and d["person"] is None and face_sync.SESSION["strictness"] == 0.03
        d = ask(env, "use the suggested threshold for Maria")
        assert d and d["action"] == "suggested" and d["person"] == 1
        d, out, _ = run(env, "from now on only make folders for people in at least 2 photos")
        assert d["action"] == "preference" and face_sync.SESSION["min_faces_for_new_person"] == 2
        assert "config.json" in out
    finally:
        face_sync.SESSION.update(strictness=0.0, min_faces_for_new_person=None)
        env.close()


# ---------------- not about the faces ----------------

def test_normal_chat_untouched():
    env = setup()
    try:
        run(env, "1 is Maria")
        for msg in ("Maria said she's coming tomorrow", "what's the weather like?", "tell me a joke",
                    "Maria is coming tomorrow"):
            d = ask(env, msg)
            assert d is None or d["action"] == "question", (msg, d)
        ff.expire_weak()
        assert ask(env, "12 is a good number") is None
        assert ask(env, "yes") is None
        d = ask(env, "where are Maria's photos?")
        assert d and d["action"] == "question"
        d = ask(env, "how many photos of Maria do I have?")
        assert d and d["action"] == "question"
        extra = ff.prompt_extra(env.cfg, "where are Maria's photos?", question=True)
        assert "You cannot see photos" in extra and "#1 Maria: 3 photos" in extra and "folder" in extra, extra
        assert "PEOPLE (from the face tool" not in ff.prompt_extra(env.cfg, "tell me a joke")
        assert "FACE TOOL" in ff.prompt_extra(env.cfg, "tell me a joke")
    finally:
        env.close()


def test_weak_window_and_music():
    env = setup(maybe=True)
    try:
        for _ in range(3):
            ask(env, "tell me a joke", fresh=False)
        assert ask(env, "all yes", fresh=False) is None      # 'yes' after 3 messages -> normal chat
        # music report, then faces report: "yes" belongs to faces
        music_followup.PENDING = {"turns_seen": 0, "items": [], "clarify": None, "confirm": None}
        r = env.run()
        ff.open_context(r, env.cfg)
        ff.open_maybe_context(faces.OPEN_MAYBE or faces.open_maybe(env.open()), env.cfg)
        music_followup.expire_weak()
        assert music_followup.PENDING["turns_seen"] > 100
        d = ask(env, "all yes", fresh=False)
        assert d and d["action"] == "answer"
        # a music report later: faces 'yes' expires, numbers still work
        ff.expire_weak()
        assert ask(env, "all yes", fresh=False) is None
        d = ask(env, "person 1 is Maria")
        assert d and d["action"] == "rename"
    finally:
        music_followup.PENDING = None
        env.close()


def test_not_followups():
    env = setup()
    try:
        assert ask(env, "/faces people") is None
        assert ask(env, "1 is Maria https://example.com") is None
        assert ask(env, f"sort the photos in {env.src} by face") is None   # a new request, not a follow-up
    finally:
        env.close()


# ---------------- the chat model (stub) ----------------

class StubLLM:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def create_chat_completion(self, messages, **kw):
        self.calls += 1
        self.prompt = messages[0]["content"]
        return {"choices": [{"message": {"content": self.text}}]}


def lines(**kw):
    base = {"ABOUT_FACES": "yes", "ACTION": "none", "PERSON": "none", "OTHER": "none", "NAME": "none", "ITEMS": "none"}
    base.update(kw)
    return "\n".join(f"{k}: {v}" for k, v in base.items())


def test_llm_stub():
    env = setup(maybe=True)
    env.cfg["faces"]["followup"]["use_llm"] = True
    try:
        msg = "number 1 is my grandma penka who lives in sofia"   # rules: too long a lowercase name -> unsure
        llm = StubLLM(lines(ACTION="rename", PERSON="1", NAME="Grandma Penka"))
        d = ask(env, msg, llm)
        assert llm.calls == 1 and d["action"] == "rename" and d["person"] == 1 and d["name"] == "Grandma Penka", d
        assert "PEOPLE" in llm.prompt and "1: Person 1" in llm.prompt and "IMG_0450.jpg" in llm.prompt
        # invalid numbers are dropped -> nothing valid left -> ask
        d = ask(env, msg, StubLLM(lines(ACTION="rename", PERSON="99", NAME="Penka")))
        assert d["action"] == "clarify", d
        # invalid name -> ask
        d = ask(env, msg, StubLLM(lines(ACTION="rename", PERSON="1", NAME="C:\\temp\\x")))
        assert d["action"] == "clarify"
        # items: numbers not on the list are dropped
        d = ask(env, msg, StubLLM(lines(ACTION="confirm", ITEMS="1, 7")))
        assert d["action"] == "answer" and d["ops"] == [(1, "yes", None)], d
        # not about faces -> normal chat
        assert ask(env, msg, StubLLM(lines(ABOUT_FACES="no"))) is None
        assert ask(env, msg, StubLLM("I think you mean something else")) is None
        # rules decided: the model is not asked
        llm = StubLLM(lines(ACTION="rename", PERSON="2", NAME="X"))
        d = ask(env, "person 2 is Ivan", llm)
        assert llm.calls == 0 and d["name"] == "Ivan"
    finally:
        env.close()


if __name__ == "__main__":
    try:
        failed = run_all(globals())
    finally:
        face_models.models = TF._REAL_MODELS
    sys.exit(1 if failed else 0)
