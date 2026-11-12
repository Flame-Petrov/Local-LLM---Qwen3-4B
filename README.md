# Local LLM

Local Python chat application with web, file, media, and photo tools.

Use Python 3.12 and a local virtual environment with the dependencies required by
the features you use. Virtual environments, model weights, and FFmpeg are not
distributed in this repository. Dependency installation on a clean machine has
not yet been verified; there is currently no dependency lockfile.

For a new checkout, create your private settings file in PowerShell:

```powershell
Copy-Item config/config.example.json config/config.json
```

Do this only if `config/config.json` does not already exist. Edit the local copy
for your machine. It uses the included `config/page_dict.example.json` rules;
you can point it at a separate local dictionary if needed. Debug logging is off
in the example. Set `BRAVE_API_KEY` in your environment if using Brave search.

Run `run.bat` from Windows, or `.venv\Scripts\python.exe src\chat.py`.
The configured chat model can download on first use; optional features may
download additional models. Keep FFmpeg in `bin/` for features that require it.

| Folder | Contents |
| --- | --- |
| `src/` | Python application modules and MCP server |
| `config/` | Public example settings and price-label rules; private local settings are ignored |
| `scripts/` | Price audit utility |
| `tests/` | Offline checks and optional real-model tests |
| `data/` | Chats, logs, documents, downloads, and audit output |
| `models/` | Local model assets |
| `bin/` | FFmpeg binary |
| `docs/PUBLISHING.md` | Complete public file list and publishing checks |
| `bluetooth-scan/` | Standalone Bluetooth utility source; device records and reports are ignored |

### Inside `src/`

| Path | Contents |
| --- | --- |
| `chat.py` | Entry point: the chat app (`run.bat` starts it) |
| `mcp_server.py` | Entry point: MCP tools for Claude Code / Desktop |
| `core/` | `config`, `debuglog`, `ui` (spinner), `live` (typing while busy), `calc` |
| `web/` | `search`, `planner`, `agent`, `prices`, `images`, `profiles` |
| `files/` | `reader` (`/read`), `writer` (`/write`) |
| `media/` | `video`, `see`, `music`, `music_followup` |
| `photos/` | `/faces` (`faces`, `faces_followup`, `face_*`), `/cutout` + `/layers` (`layers`, `layer_ops`, `layer_models`) |

Each package folder has its own `README.md` that explains why its files are grouped together.
Imports are absolute from `src/`: `from core import config`, `from media.music import ...`.
Module self-checks run as modules from `src/`, e.g. `..\.venv\Scripts\python.exe -m photos.layer_models --selfcheck <photo>`.

Paths in `config/config.json` resolve relative to the repository root.
Run the audit with `.venv\Scripts\python.exe scripts\audit.py` (uses the network and local model).
Run the MCP server with `.venv\Scripts\python.exe src\mcp_server.py`.
Existing external MCP registrations should point to this new script path.

Run an offline test with `.venv\Scripts\python.exe tests\test_calc.py`;
other `tests/test_*.py` scripts use the same convention. The real face-model
smoke test requires `FACES_SAMPLE` and may download model files.

Git uses an explicit file allowlist. Runtime data, saved or loaded chats and
memory, logs, downloads, models, binaries, environments, local configuration,
planning notes, screenshots, and historical inventories stay local. See
[the publishing inventory](docs/PUBLISHING.md) before staging or pushing.
