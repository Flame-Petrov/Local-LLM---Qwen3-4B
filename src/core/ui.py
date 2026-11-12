"""
Animated "thinking" indicator for the terminal. All settings come from config.json -> "ui".

    ⠹ Pondering… 7s

Runs in a background thread. llama.cpp releases Python's GIL while it computes,
so the animation stays smooth even while the model is busy.
Automatically disabled when output isn't an interactive terminal (pipes/logs).

The spinner writes to its OWN duplicate of the console handle, not sys.stdout:
llama-cpp-python (verbose=False) temporarily redirects sys.stdout and fd 1 to
os.devnull while loading the model, which would otherwise swallow every frame
and make the animation look frozen.
"""

import atexit
import os
import shutil
import sys
import textwrap
import threading
import time
import unicodedata

# Terminal control sequences (protocol constants, not settings).
RESET = "\x1b[0m"
CLEAR_LINE = "\r\x1b[2K"
HIDE_CURSOR, SHOW_CURSOR = "\x1b[?25l", "\x1b[?25h"


def _color(n: int) -> str:
    return f"\x1b[38;5;{n}m"


def _bg(n: int) -> str:
    return f"\x1b[48;5;{n}m"


_ansi_on = False

# ---- Ctrl+C tracking: when (if at all) the current turn was cancelled ----
_cancelled_at = None


def reset_cancel():
    global _cancelled_at
    _cancelled_at = None


def note_cancel():
    """Record the moment Ctrl+C stopped the bot (first press wins)."""
    global _cancelled_at
    if _cancelled_at is None:
        _cancelled_at = time.monotonic()


def cancel_time():
    return _cancelled_at


# ---- editable pre-filled input (Windows console) ----
def _push_console_input(text: str):
    """Type `text` into the console's input buffer as key presses, so the next
    input() shows it already typed and fully editable (arrows, Backspace, Home/End)."""
    import ctypes
    from ctypes import wintypes

    class KEY_EVENT_RECORD(ctypes.Structure):
        _fields_ = [("bKeyDown", wintypes.BOOL), ("wRepeatCount", wintypes.WORD),
                    ("wVirtualKeyCode", wintypes.WORD), ("wVirtualScanCode", wintypes.WORD),
                    ("uChar", wintypes.WCHAR), ("dwControlKeyState", wintypes.DWORD)]

    class _EVENT(ctypes.Union):
        _fields_ = [("KeyEvent", KEY_EVENT_RECORD), ("_pad", ctypes.c_byte * 16)]

    class INPUT_RECORD(ctypes.Structure):
        _fields_ = [("EventType", wintypes.WORD), ("Event", _EVENT)]

    KEY_EVENT = 0x0001
    units = text.encode("utf-16-le")
    chars = [units[i:i + 2].decode("utf-16-le", "surrogatepass") for i in range(0, len(units), 2)]
    records = (INPUT_RECORD * (2 * len(chars)))()
    for i, ch in enumerate(chars):
        for j, down in enumerate((True, False)):
            r = records[2 * i + j]
            r.EventType = KEY_EVENT
            r.Event.KeyEvent.bKeyDown = down
            r.Event.KeyEvent.wRepeatCount = 1
            r.Event.KeyEvent.uChar = ch
    k32 = ctypes.windll.kernel32
    written = wintypes.DWORD()
    k32.WriteConsoleInputW(k32.GetStdHandle(-10), records, len(records), ctypes.byref(written))


def input_prefilled(prompt: str, text: str = "") -> str:
    """input() whose line starts with `text` already typed (editable) when possible."""
    if text and os.name == "nt" and sys.stdin.isatty():
        try:
            _push_console_input(text)
        except Exception:
            pass  # fall back to an empty line
    return input(prompt)


def enable_ansi():
    global _ansi_on
    if os.name == "nt" and not _ansi_on:
        os.system("")  # switches the Windows console into ANSI/VT mode
    _ansi_on = True


def clear_screen() -> bool:
    """Wipe the screen AND the scrollback, cursor to the top. Does nothing when stdout is not a
    terminal (pipes, tests). Returns True if it cleared."""
    try:
        if not sys.stdout.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    try:
        enable_ansi()
        # H = cursor home, 2J = clear screen, 3J = clear scrollback. 3J must come LAST: in
        # Windows Terminal / VS Code, 2J pushes the visible screen into the scrollback.
        sys.stdout.write("\033[H\033[2J\033[3J")
        sys.stdout.flush()
    except Exception:
        if os.name == "nt":
            os.system("cls")
        else:
            return False
    return True


def colors_on() -> bool:
    """ANSI colours only on a real terminal - never in pipes, logs or the chat history."""
    try:
        on = sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False
    if on:
        enable_ansi()
    return on


def paint(text: str, color_n) -> str:
    """`text` in 256-colour `color_n` (None = as is). Plain text when output is not a terminal."""
    if color_n is None or not text or not colors_on():
        return text
    return _color(color_n) + text + RESET


def file_link(path) -> str:
    """The path as an OSC 8 hyperlink (Ctrl+click opens it in Windows Terminal); the visible text
    stays the plain path. Plain path when output is not a terminal."""
    s = str(path)
    if not colors_on():
        return s
    try:
        from pathlib import Path
        uri = Path(s).resolve().as_uri()
    except (ValueError, OSError):
        return s
    return f"\x1b]8;;{uri}\x1b\\{s}\x1b]8;;\x1b\\"


def display_width(s: str) -> int:
    """Terminal columns a string occupies (wide CJK/emoji count as 2)."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def echo_user_message(prompt: str, message: str, mcfg: dict, replace_typed: bool = True):
    """Replace the line the user just typed with a full-width highlighted block,
    like Claude CLI, so each question is easy to spot above its answer.
    replace_typed=False just prints the block (for a message sent from the live footer)."""
    if not (mcfg["highlight"] and sys.stdout.isatty() and sys.stdin.isatty()):
        return
    enable_ansi()
    cols = shutil.get_terminal_size().columns
    typed_rows = max(1, -(-display_width(prompt + message) // cols)) if replace_typed else 0
    style = _bg(mcfg["background_256"]) + _color(mcfg["foreground_256"])
    lines = textwrap.wrap(mcfg["prefix"] + message, width=max(10, cols - 1),
                          subsequent_indent=" " * len(mcfg["prefix"]),
                          drop_whitespace=False) or [mcfg["prefix"]]
    out = f"\x1b[{typed_rows}A\r\x1b[J" if typed_rows else ""  # back up over the typed input and clear it
    for line in lines:
        out += style + line + "\x1b[K" + RESET + "\n"  # \x1b[K paints bg to the right edge
    if mcfg["blank_line_after"]:
        out += "\n"
    sys.stdout.write(out)
    sys.stdout.flush()


class Spinner:
    def __init__(self, ucfg: dict):
        self.cfg = ucfg
        self.enabled = ucfg["spinner"] and sys.stdout.isatty()
        utf8 = (sys.stdout.encoding or "").lower().startswith("utf")
        self.frames = ucfg["frames_unicode"] if utf8 else ucfg["frames_ascii"]
        self.ellipsis = "…" if utf8 else "..."
        c = ucfg["colors_256"]
        self.c_spin, self.c_dim, self.c_mid, self.c_bright = (
            _color(c["spinner"]), _color(c["dim"]), _color(c["mid"]), _color(c["bright"]))
        self._label = None  # None = rotate through thinking_words
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._t0 = self._words_t0 = 0.0
        self._inline = False  # True = small spinner at the cursor, mid-answer
        self.live = None      # live.LiveInput: while its footer is shown, the spinner draws there
        self.out = sys.stdout
        if self.enabled:
            enable_ansi()
            # Private handle to the console that later stdout redirects can't touch.
            self.out = open(os.dup(sys.stdout.fileno()), "w", encoding="utf-8",
                            errors="replace", closefd=True)
            atexit.register(self._write, SHOW_CURSOR)

    def _write(self, s: str):
        self.out.write(s)
        self.out.flush()

    @property
    def running(self) -> bool:
        return self._thread is not None

    @property
    def footer(self) -> bool:
        """True while the live-input footer is shown: the spinner lives in its status row."""
        return bool(self.live and self.live.active)

    def start(self, label: str | None = None):
        """Start animating, or just change the label if already running."""
        if not self.enabled:
            return
        self.set(label)
        if self.running:
            return
        self._t0 = self._words_t0 = time.monotonic()
        self._stop.clear()
        sys.stdout.flush()  # anything already printed must land before the spinner draws
        self._write(HIDE_CURSOR)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def start_inline(self):
        """Small spinner right after the text already on screen (no line clearing).
        Used while a long answer resumes after its context was trimmed."""
        if not self.enabled or self.running:
            return
        if self.footer:  # the footer status row shows progress - nothing is drawn in the text
            self.start()
            return
        self._inline = True
        self._stop.clear()
        sys.stdout.flush()
        self._write(HIDE_CURSOR + "  ")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def status(self, key: str, **values):
        """Start/switch to a named status label from config ui.status (e.g. 'reading', site=...)."""
        self.start(self.cfg["status"][key].format(**values))

    def set(self, label: str | None):
        with self._lock:
            if label is None and self._label is not None:
                self._words_t0 = time.monotonic()  # restart rotation at the first word
            self._label = label

    def log(self, message: str):
        """Print a permanent line above the spinner without breaking the animation."""
        if not self.running or self.footer:
            print(message)
            return
        with self._lock:
            self._write(CLEAR_LINE + message + "\n")

    def stop(self):
        if not self.running:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        if self.footer:
            self.live.set_status("")
            self._inline = False
            return
        if self._inline:
            self._inline = False
            self._write("\b\b  \b\b" + SHOW_CURSOR)  # erase the 2 cells it used
        else:
            self._write(CLEAR_LINE + SHOW_CURSOR)

    # ---- drawing ----
    def _shimmer(self, text: str, tick: int) -> str:
        """Bright band sweeping left-to-right across the text."""
        rb, rm = self.cfg["shimmer_bright_radius"], self.cfg["shimmer_mid_radius"]
        width = len(text) + 2 * (rm + 1)
        center = tick % width - (rm + 1)
        out = []
        for i, ch in enumerate(text):
            d = abs(i - center)
            out.append((self.c_bright if d <= rb else self.c_mid if d <= rm else self.c_dim) + ch)
        return "".join(out) + RESET

    def _run(self):
        words = self.cfg["thinking_words"]
        tick = 0
        while not self._stop.is_set():
            now = time.monotonic()
            elapsed = now - self._t0
            if self._inline:
                self._write(f"\b{self.c_spin}{self.frames[tick % len(self.frames)]}{RESET}")
                tick += 1
                self._stop.wait(self.cfg["frame_interval_s"])
                continue
            with self._lock:
                word = words[int((now - self._words_t0) // self.cfg["word_change_s"]) % len(words)]
                label = self._label or word
                frame = self.frames[tick % len(self.frames)]
                timer = (f" {self.c_dim}{int(elapsed)}s{RESET}"
                         if elapsed >= self.cfg["show_timer_after_s"] else "")
                line = f"{self.c_spin}{frame}{RESET} {self._shimmer(label + self.ellipsis, tick)}{timer}"
            if self.footer:
                self.live.set_status(line)
            else:
                from core.live import fit_width  # never wrap: a wrapped spinner row can't be cleared
                with self._lock:
                    self._write(CLEAR_LINE + fit_width(line, shutil.get_terminal_size().columns - 1))
            tick += 1
            self._stop.wait(self.cfg["frame_interval_s"])
