"""
Live input: type instructions while the bot is working.

While a turn runs, two lines are pinned to the bottom of the terminal (a VT scroll
region keeps the answer scrolling above them):

    ⠹ Analyzing what it found… 34s   ⏳ 1 waiting: "only under 10 €"
    › also check emag█

- Enter sends the line. The message is QUEUED (⏳ waiting) until the bot reaches a point
  where it can use it; while the model reads it the indicator shows 📥 reading, and once
  the model answers with it in mind: ✓ taken into account.
- If the model is in the middle of a call that can be restarted (an answer, or the agent's
  analysis), the call is interrupted within ~2 s and restarted with the new message added
  at the END of the prompt - so llama.cpp reuses everything it already read (fast).
- Ctrl+C still stops everything as before.

All settings: config.json -> "live_input". Windows console only (msvcrt); elsewhere it
is simply off and the chat works as before.
"""

import _thread
import os
import re
import shutil
import signal
import sys
import threading
import time
from contextlib import contextmanager

try:
    import msvcrt
except ImportError:  # not Windows
    msvcrt = None

from core.debuglog import log

RESET = "\x1b[0m"


class NewInput(KeyboardInterrupt):
    """Raised in the main thread when the user sends a message while the model is busy.
    Subclass of KeyboardInterrupt so that any code path that does not know about live
    input still stops safely instead of crashing."""


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def fit_width(s: str, cols: int) -> str:
    """Cut a coloured line to `cols` visible columns (escape codes kept, wide chars = 2),
    so it never wraps: a wrapped footer/spinner row leaves junk behind and can scroll the screen."""
    import unicodedata
    out, used, i = [], 0, 0
    while i < len(s):
        m = _ANSI.match(s, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        w = 2 if unicodedata.east_asian_width(s[i]) in "WF" else 1
        if used + w > cols:
            break
        out.append(s[i])
        used += w
        i += 1
    return "".join(out)


def _cursor_pos():
    """(row, col), 1-based, of the console cursor inside the visible window - or None.
    Asked from the Windows console API, because the VT way (ESC[6n) answers through stdin,
    which the key-reading thread owns."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class CSBI(ctypes.Structure):
            _fields_ = [("dwSize", wintypes._COORD), ("dwCursorPosition", wintypes._COORD),
                        ("wAttributes", wintypes.WORD), ("srWindow", wintypes.SMALL_RECT),
                        ("dwMaximumWindowSize", wintypes._COORD)]
        info = CSBI()
        h = ctypes.windll.kernel32.GetStdHandle(-11)
        if not ctypes.windll.kernel32.GetConsoleScreenBufferInfo(h, ctypes.byref(info)):
            return None
        return (info.dwCursorPosition.Y - info.srWindow.Top + 1, info.dwCursorPosition.X - info.srWindow.Left + 1)
    except Exception:
        return None


def _console_size(stream):
    """(rows, cols) of the visible console window, asked through `stream`'s own OS handle -
    unaffected when a library redirects fd 1 / the std handle. None if it is not a console."""
    if os.name != "nt" or msvcrt is None:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class CSBI(ctypes.Structure):
            _fields_ = [("dwSize", wintypes._COORD), ("dwCursorPosition", wintypes._COORD),
                        ("wAttributes", wintypes.WORD), ("srWindow", wintypes.SMALL_RECT),
                        ("dwMaximumWindowSize", wintypes._COORD)]
        info = CSBI()
        h = msvcrt.get_osfhandle(stream.fileno())
        if not ctypes.windll.kernel32.GetConsoleScreenBufferInfo(ctypes.c_void_p(h), ctypes.byref(info)):
            return None
        w = info.srWindow
        return w.Bottom - w.Top + 1, w.Right - w.Left + 1
    except Exception:
        return None


def _fg(n):
    return f"\x1b[38;5;{n}m"


def _bg(n):
    return f"\x1b[48;5;{n}m"


class _LockedWriter:
    """sys.stdout replacement while the footer is shown: every write is atomic with
    respect to footer redraws (which jump to the bottom rows and back)."""

    def __init__(self, real, lock):
        self._real, self._lock = real, lock

    def write(self, s):
        with self._lock:
            n = self._real.write(s)
            self._real.flush()
            return n

    def flush(self):
        with self._lock:
            self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


class LiveInput:
    def __init__(self, lcfg: dict, out):
        self.cfg = lcfg
        self.out = out  # the spinner's private console handle (survives library stdout redirects)
        self.enabled = bool(lcfg["enabled"] and msvcrt and sys.stdin.isatty() and sys.stdout.isatty())
        utf8 = (getattr(out, "encoding", "") or "").lower().startswith("utf")
        self.sym = lcfg["symbols"] if utf8 else lcfg["symbols_ascii"]
        self.lock = threading.RLock()
        self.active = False
        self.draft, self.cursor = "", 0
        self.queue = []        # sent, not yet taken: [{"text", "t"}]
        self.reading = []      # taken into a prompt, model still reading
        self.used = []         # (text, time) answered with - shown for a few seconds
        self.consumed_turn = []  # every message taken this turn (goes into chat history)
        self.status = ""
        self.interruptible = False
        self.interrupt_requested = False
        self._thread = None
        self._size = (0, 0)
        self._orig_stdout = None

    # ---------------- Ctrl+C / interrupt plumbing ----------------
    def install_sigint(self):
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, signum, frame):
        if self.interrupt_requested:  # our own wake-up call (a new message), not the user's Ctrl+C
            self.interrupt_requested = False
            if self.interruptible:
                raise NewInput()
            return  # the model call already finished: the message is picked up at the next step
        raise KeyboardInterrupt

    @contextmanager
    def interruptible_section(self):
        """Model calls inside may be interrupted (NewInput) when the user sends a message."""
        prev = self.interruptible
        self.interruptible = True
        try:
            yield
        finally:
            self.interruptible = prev

    # ---------------- queue ----------------
    def is_stop(self, texts) -> bool:
        """Does the user ask to stop ('stop', 'enough', 'спри', 'стига', ...)?"""
        return any(re.search(p, t, re.I) for t in texts for p in self.cfg["stop_intent_patterns"])

    def take(self) -> list[str]:
        """All waiting messages -> 'reading' state. Returns their texts.
        Commands ('/exit', '/load', ...) are not instructions: they stay queued and run after the turn."""
        with self.lock:
            texts = [q["text"] for q in self.queue if not q["cmd"]]
            self.queue = [q for q in self.queue if q["cmd"]]
            self.reading += texts
            self.consumed_turn += texts
        if texts:
            log("live_input_taken", texts=texts)
            self._draw()
        return texts

    def take_if(self, pred) -> list[str]:
        """Like take(), but only the waiting messages `pred(text)` accepts. The others stay queued
        and run as the next turn (e.g. a question typed during a music download)."""
        with self.lock:
            texts, keep = [], []
            for q in self.queue:
                try:
                    ok = not q["cmd"] and bool(pred(q["text"]))
                except Exception:
                    ok = False
                if ok:
                    texts.append(q["text"])
                else:
                    keep.append(q)
            self.queue = keep
            self.reading += texts
            self.consumed_turn += texts
        if texts:
            log("live_input_taken", texts=texts, filtered=True)
            self._draw()
        return texts

    def mark_used(self, texts):
        if not texts:
            return
        with self.lock:
            now = time.monotonic()
            for t in texts:
                if t in self.reading:
                    self.reading.remove(t)
                self.used.append((t, now))
        log("live_input_used", texts=list(texts))
        self._draw()

    def pop_consumed(self) -> list[str]:
        with self.lock:
            out, self.consumed_turn = self.consumed_turn, []
            self.reading.clear()
        return out

    def pending_texts(self) -> list[str]:
        with self.lock:
            out = [q["text"] for q in self.queue]
            self.queue.clear()
        return out

    # ---------------- footer on / off ----------------
    def start(self):
        if not self.enabled or self.active:
            return
        sys.stdout.flush()
        rows, cols = self._term_size()
        with self.lock:
            # make room for the 2 footer rows below the cursor, then confine scrolling above them
            self._w("\n\n\x1b[2A\x1b7" + f"\x1b[1;{rows - 2}r" + "\x1b8\x1b[?25l")
            self._size = (rows, cols)
            self.active = True
            self._orig_stdout = sys.stdout
            sys.stdout = _LockedWriter(sys.stdout, self.lock)
        self._thread = threading.Thread(target=self._keys, daemon=True)
        self._thread.start()
        self._draw()

    def stop(self) -> str:
        """Remove the footer. Returns the unsent draft (to pre-fill the normal input line)."""
        if not self.active:
            return ""
        self.active = False
        if self._thread:
            self._thread.join()
            self._thread = None
        with self.lock:
            rows, _ = self._size
            sys.stdout = self._orig_stdout
            self._w(f"\x1b7\x1b[{rows - 1};1H\x1b[2K\x1b[{rows};1H\x1b[2K\x1b[r\x1b8\x1b[?25h")
            draft, self.draft, self.cursor = self.draft, "", 0
            self.used.clear()
        return draft

    # ---------------- drawing ----------------
    def set_status(self, line: str):
        self.status = line
        self._draw()

    def _w(self, s):
        self.out.write(s)
        self.out.flush()

    def _term_size(self):
        # Ask the footer's own console handle (a dup made at startup), NOT fd 1: libraries such as the
        # vision loader redirect fd 1 to devnull, and shutil then silently returns 80x24, which moved the
        # footer to the middle of the screen. If even that fails, keep the last good size.
        win = _console_size(self.out)
        if win:
            return max(win[0], 5), max(win[1], 20)
        try:
            size = os.get_terminal_size(sys.__stdout__.fileno())
        except (OSError, ValueError, AttributeError):
            if self._size != (0, 0):
                return self._size
            size = shutil.get_terminal_size()
        return max(size.lines, 5), max(size.columns, 20)

    def _preview(self, text):
        n = self.cfg["preview_chars"]
        return text if len(text) <= n else text[: n - 1] + "…"

    def _indicator(self):
        c, T, S = self.cfg["colors_256"], self.cfg["texts"], self.sym
        now = time.monotonic()
        self.used = [(t, ts) for t, ts in self.used if now - ts < self.cfg["used_show_s"]]
        if self.queue:
            last = self.queue[-1]
            key = "waiting_command" if last["cmd"] else "waiting"
            return (_fg(c["waiting"]) + f"{S['waiting']} "
                    + T[key].format(n=len(self.queue), text=self._preview(last["text"])))
        if self.reading:
            return _fg(c["reading"]) + f"{S['reading']} " + T["reading"].format(text=self._preview(self.reading[-1]))
        if self.used:
            return _fg(c["used"]) + f"{S['used']} " + T["used"].format(text=self._preview(self.used[-1][0]))
        return _fg(c["dim"]) + T["hint"]

    def _draw(self):
        if not self.active:
            return
        with self.lock:
            rows, cols = self._term_size()
            if (rows, cols) != self._size:
                self._relayout(rows)
                self._size = (rows, cols)
            c = self.cfg["colors_256"]
            # keep the row's background after colour resets inside the spinner text
            status = f"{self.status}   {self._indicator()}".replace(RESET, RESET + _bg(c["status_bg"]))
            status = fit_width(status, cols - 1)
            prompt = self.cfg["prompt"]
            room = cols - len(prompt) - 2
            start = max(0, self.cursor - room)
            shown = self.draft[start: start + room]
            cur = self.cursor - start
            caret = shown[cur] if cur < len(shown) else " "
            line = shown[:cur] + "\x1b[7m" + caret + "\x1b[27m" + shown[cur + 1:]
            self._w("\x1b7"
                    + f"\x1b[{rows - 1};1H{_bg(c['status_bg'])}\x1b[2K{status}{RESET}"
                    + f"\x1b[{rows};1H{_bg(c['input_bg'])}{_fg(c['text'])}\x1b[2K{prompt}{line}{RESET}"
                    + "\x1b8")

    def _relayout(self, rows):
        """Terminal resized. The old footer rows are NOT moved by the terminal: they stay where
        they were (mid-screen when the window grew) and new output would be printed over them.
        So: erase everything below the text cursor (only the old footer lives there), make sure
        the cursor is above the 2 new footer rows, and move the scroll region. Caller holds the lock."""
        pos = _cursor_pos()
        if pos is None:  # no console API: clear below the saved text position, move the region
            self._w(f"\x1b7\x1b[r\x1b8\x1b[J\x1b7\x1b[1;{rows - 2}r\x1b8")
            return
        row, col = pos
        seq = f"\x1b[r\x1b[{row};{col}H\x1b[J"  # full-screen region, back to the text cursor, clear below
        if row > rows - 2:  # window got shorter: the cursor would sit in the footer -> scroll text up
            k = row - (rows - 2)
            seq += f"\x1b[{rows};1H" + "\n" * k + f"\x1b[{rows - k + 1};1H\x1b[J"
            row -= k
        seq += f"\x1b[1;{rows - 2}r\x1b[{row};{col}H"  # (setting the region homes the cursor)
        self._w(seq)
        log("live_footer_relayout", rows=rows, cursor_row=row, cursor_col=col)

    # ---------------- keyboard ----------------
    def _keys(self):
        while self.active:
            if not msvcrt.kbhit():
                # resizes must be handled even when nothing else redraws (e.g. while an answer streams)
                if self._term_size() != self._size:
                    self._draw()
                time.sleep(self.cfg["poll_s"])
                continue
            ch = msvcrt.getwch()
            with self.lock:
                if ch in ("\x00", "\xe0"):  # arrows / Home / End / Delete
                    code = msvcrt.getwch()
                    if code == "K":
                        self.cursor = max(0, self.cursor - 1)
                    elif code == "M":
                        self.cursor = min(len(self.draft), self.cursor + 1)
                    elif code == "G":
                        self.cursor = 0
                    elif code == "O":
                        self.cursor = len(self.draft)
                    elif code == "S":
                        self.draft = self.draft[: self.cursor] + self.draft[self.cursor + 1:]
                elif ch == "\r":
                    self._submit()
                elif ch == "\x08":
                    if self.cursor:
                        self.draft = self.draft[: self.cursor - 1] + self.draft[self.cursor:]
                        self.cursor -= 1
                elif ch == "\x1b":
                    self.draft, self.cursor = "", 0
                elif ch >= " ":
                    self.draft = self.draft[: self.cursor] + ch + self.draft[self.cursor:]
                    self.cursor += 1
            self._draw()

    def _submit(self):
        text = self.draft.strip()
        self.draft, self.cursor = "", 0
        if not text:
            return
        is_cmd = text.startswith("/")
        self.queue.append({"text": text, "t": time.monotonic(), "cmd": is_cmd})
        log("live_input_received", text=text, command=is_cmd, model_busy=self.interruptible)
        if self.cfg["interrupt_model"] and self.interruptible and not is_cmd:
            self.interrupt_requested = True
            _thread.interrupt_main(signal.SIGINT)  # wakes the main thread between model steps


# ---------------- module-level access (no-ops when live input is off) ----------------
LIVE: LiveInput | None = None


def take() -> list[str]:
    return LIVE.take() if LIVE else []


def take_if(pred) -> list[str]:
    return LIVE.take_if(pred) if LIVE else []


def is_stop(texts) -> bool:
    return bool(LIVE and texts and LIVE.is_stop(texts))


def mark_used(texts):
    if LIVE:
        LIVE.mark_used(texts)


@contextmanager
def interruptible():
    if LIVE:
        with LIVE.interruptible_section():
            yield
    else:
        yield


def symbol(name: str) -> str:
    return LIVE.sym[name] if LIVE else ""


def text_for(name: str, **values) -> str:
    return LIVE.cfg["texts"][name].format(**values) if LIVE else ""
