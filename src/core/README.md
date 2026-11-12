# core/

Shared infrastructure that the rest of the app is built on. Nothing here is a user feature on its own.
These modules sit at the bottom of the import graph: every other package imports from `core`, and
`core` imports only from itself.

| File | What it does | Who uses it |
| --- | --- | --- |
| `config.py` | Loads `config/config.json`, which holds every prompt, rule and setting. `config.path()` resolves data paths against the project root. | Almost every module |
| `debuglog.py` | Writes the debug log: one JSON Lines file per run (`clip`, `log`). | Almost every module |
| `ui.py` | The animated "thinking" spinner and terminal echo helpers. | `chat`, `web/agent`, `media/music*`, `photos/faces`, `photos/face_sync`, `photos/layers` |
| `live.py` | Live input: lets you type and queue instructions while the bot is working. | `chat`, `web/agent`, `media/music_followup`, `photos/faces` |
| `calc.py` | Calculator tool for the model, plus a check of the arithmetic in answers. | `chat` |

**Why they're together:** these are the cross-cutting pieces (settings, logging, the terminal) that
any feature may need. Keeping them in one leaf package means a feature package never has to import
another feature package just to log something or read a setting.

`calc.py` is the odd one out: it's a small model tool, not infrastructure. It lives here because it's
pure, self-contained and too small to need a package of its own.

`config.py` locates the project root from its own position (`parents[2]`). If you move it, update
`ROOT`.
