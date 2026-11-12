# Local LLM

Run a Qwen3-based model locally from a Windows terminal, with optional web search,
document, music, video, and photo tools. No Ollama or model server is required.
The repository contains source code and example settings; **model weights and
installed dependencies are not included**.

## 1. Prerequisites

These instructions target **64-bit Windows and Python 3.12**, matching the local
development environment. Other operating systems have not been validated; the
launcher and some terminal/Bluetooth functionality are Windows-specific.

- Install [Python 3.12 for Windows](https://www.python.org/downloads/windows/),
  including the Python launcher (`py`). Check with `py -3.12 --version`.
- Use PowerShell in the extracted repository folder. All commands below run there.
- Have internet access for installing packages and the first use of each model.
- Allow about **2.5 GB for the chat model alone**, plus space for Python packages,
  optional models, and your output. Leave several additional GB free.
- CPU inference is the default; a dedicated GPU is not required. As a practical
  starting point, aim for 8 GB RAM for chat and 16 GB or more for combined photo/
  video workflows. These are planning estimates, not tested minimums; memory use
  depends on context length and which models are loaded.

## 2. Download the project

Either use GitHub's **Code > Download ZIP**, extract it completely, and open
PowerShell inside the extracted folder, or use Git:

```powershell
git clone https://github.com/Flame-Petrov/Local-LLM---Qwen3-4B.git
cd Local-LLM---Qwen3-4B
```

You should see `run.bat`, `src`, and `config` in this folder. Do not run the app
from inside the ZIP archive.

## 3. Create the environment and install the chat dependencies

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "llama-cpp-python==0.3.36" --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
.\.venv\Scripts\python.exe -m pip install requests huggingface-hub numpy opencv-python-headless pillow
```

The CPU wheel avoids compiling llama.cpp locally. Version `0.3.36` is the version
used in the existing environment. If no compatible wheel is available, consult
[llama-cpp-python's installation instructions](https://github.com/abetlen/llama-cpp-python)
for supported wheels or a source build. Building from source on Windows requires
a C/C++ toolchain such as Visual Studio Build Tools with C++ support; installing
Python alone does not provide it.

NumPy and OpenCV are needed even for plain chat because photo modules are imported
at startup. Install `opencv-python-headless`, not multiple OpenCV variants in the
same environment. Using the environment's Python explicitly means you do not need
to activate it or change PowerShell's execution policy.

These are direct dependency installation commands, not a complete version lock.
A fresh-machine install of all features has not yet been validated.

## 4. Create your settings

```powershell
if (-not (Test-Path config/config.json)) {
    Copy-Item config/config.example.json config/config.json
}
```

Edit `config/config.json` for permanent changes. Leave the example file unchanged
so it remains a clean starting point. The example already points to the included
`config/page_dict.example.json`; no separate price dictionary download is needed.
Paths in the settings resolve relative to the repository root.

## 5. Start chatting

```powershell
.\run.bat
```

You can also double-click `run.bat`, or run Python directly for clearer errors:

```powershell
.\.venv\Scripts\python.exe src\chat.py
```

On first start the app downloads this file from Hugging Face into `models/`:

- Repository: `mradermacher/Huihui-Qwen3-4B-Instruct-2507-abliterated-GGUF`
- File: `Huihui-Qwen3-4B-Instruct-2507-abliterated.Q4_K_M.gguf`

This is the configured Huihui Qwen3 derivative, not the original unmodified Qwen3
release. Wait for the download and model loading to finish, then type a message.
Later launches reuse the local file. **`run.bat` starts the app; it does not install
Python or missing Python packages.**

To download the chat weight ahead of time, after installing the dependencies:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='mradermacher/Huihui-Qwen3-4B-Instruct-2507-abliterated-GGUF', filename='Huihui-Qwen3-4B-Instruct-2507-abliterated.Q4_K_M.gguf', local_dir='models')"
```

To use an already downloaded compatible GGUF instead:

```powershell
.\.venv\Scripts\python.exe src\chat.py --model "C:\Models\your-model.gguf"
```

That skips the chat-model download. For offline use, install packages and obtain
all models needed by your chosen features beforehand. Plain chat then runs locally;
web searches, URL fetches, and media downloads still require internet access.

Useful commands inside the chat:

| Command | Action |
| --- | --- |
| `/help` | Show every feature and its options |
| `/exit` | Quit; the conversation is saved locally |
| `/reset` | Start a new chat; previous saved chats remain |
| `/load` | List and reopen saved chats |
| `/settings` | Show current settings |
| `/reload` | Reload the settings file and clear the current conversation |
| `/auto off` | Keep automatic web search off (the example default) |

Press Ctrl+C to interrupt work. For lower memory use or a simpler terminal display:

```powershell
.\.venv\Scripts\python.exe src\chat.py --ctx 4096 --threads 4 --no-spinner
```

Choose a thread count appropriate for your CPU. See all startup options with
`.\.venv\Scripts\python.exe src\chat.py --help`.

## 6. Optional features

Install only the groups you plan to use. Run these commands in PowerShell, not
inside the chat prompt, and restart the app after installing packages.

### Web search and page reading

```powershell
.\.venv\Scripts\python.exe -m pip install ddgs trafilatura lxml
```

Use `/web <question>`, `/agent <question>`, or `/images <page URL>`. Search works
without an API key through DDGS. For optional Brave search, set `BRAVE_API_KEY`
in the same PowerShell window before starting the app:

```powershell
$env:BRAVE_API_KEY = "YOUR_BRAVE_SEARCH_API_KEY"
.\.venv\Scripts\python.exe src\chat.py
```

The example above sets it for that terminal session. Keep real keys out of source,
example settings, and commits.

### Reading and writing documents

```powershell
.\.venv\Scripts\python.exe -m pip install python-docx python-pptx openpyxl pypdf fpdf2 PyYAML lxml trafilatura charset-normalizer
```

Use `/read "C:\Documents\report.pdf" summarize this` or `/write summary.docx`.
Bare input filenames are searched in `data/documents/` and the current folder.
Generated documents go to `data/documents/` by default. Text-based PDFs are
supported; scanned PDFs need OCR outside this app. Old binary Office files such
as `.doc` and `.xls` should be converted to supported formats first.

### Music and video

```powershell
.\.venv\Scripts\python.exe -m pip install "yt-dlp[default]" imageio-ffmpeg mutagen faster-whisper
```

Install a current [Node.js LTS release](https://nodejs.org/en/download), reopen
PowerShell, and check `node --version`. The example sets `video.js_runtime` to
`node`; the app does not install Node.js. YouTube extraction also uses yt-dlp's
EJS support, supplied by the `default` extra. See the
[official yt-dlp EJS guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS) if runtime
requirements or YouTube challenges change.

FFmpeg is resolved automatically: the app first checks `PATH`, then `bin/ffmpeg.exe`,
and otherwise copies the executable supplied by `imageio-ffmpeg` into `bin/`.
Supported Windows wheels include that executable; see the
[imageio-ffmpeg documentation](https://github.com/imageio/imageio-ffmpeg).
You can also install FFmpeg separately and put it on `PATH`.

Use `/music <YouTube URL>` or `/video <URL or local file> <question>`.
Whisper's `small` speech model downloads on demand to `models/whisper/` when
transcription is needed. Video frame analysis uses the separate vision models
listed below. Site restrictions and authentication can affect downloads;
`video.cookies_from_browser` is empty by default and is an optional local setting.

### Image understanding, faces, cutouts, and layers

```powershell
.\.venv\Scripts\python.exe -m pip install onnxruntime psd-tools
```

The base setup already supplies Pillow, OpenCV, NumPy, Hugging Face downloads,
and llama-cpp-python. For optional HEIC/HEIF image support also install:

```powershell
.\.venv\Scripts\python.exe -m pip install pillow-heif
```

| Feature | First-use downloads | Local destination |
| --- | --- | --- |
| `/see <photo>`, video frame descriptions, optional cutout descriptions | Qwen2.5-VL-3B Q4_K_M GGUF plus its Q8_0 vision projector | `models/` |
| `/faces <folder or photo>` | YuNet detector and the configured ArcFace embedder (or SFace if selected) | `models/faces/` |
| `/cutout <photo>` | Configured BiRefNet matting model and any models needed by the selected options | `models/layers/` |
| `/layers <photo>` | Configured matting, depth, and segmentation models; LaMa when requested for filling | `models/layers/` |

These are additional downloads, separate from the chat model. File names, model
repositories, and supported model choices are under `video.vision`, `faces.models`,
and `layers.models` in the settings. Subsequent uses reuse downloaded files.
Face/layer model files are checked against configured hashes. Refer to each
model provider's license before redistributing weights or using them commercially.

### MCP server (optional, separate from the chat interface)

The source uses the MCP 2.x API. The existing environment uses these versions:

```powershell
.\.venv\Scripts\python.exe -m pip install "mcp==2.3.0" "mcp-types==2.3.0" anyio
.\.venv\Scripts\python.exe src\mcp_server.py --help
```

Install the photo dependencies above too. Configure your MCP client to launch the
absolute path to `.venv\Scripts\python.exe` with the absolute path to
`src\mcp_server.py` as its argument. This is a stdio server for photo tools, not
an HTTP server or a chat window. It does not need the chat GGUF just to start.
To preload and check the configured layer models before connecting a client:

```powershell
.\.venv\Scripts\python.exe src\mcp_server.py --download-models
```

### Bluetooth utilities (optional, Windows only)

```powershell
.\.venv\Scripts\python.exe -m pip install bleak
.\.venv\Scripts\python.exe bluetooth-scan\scan.py
```

Enable Bluetooth first. The scan takes approximately 90 seconds and creates local
`bluetooth-scan/results.json`. The separate `report.py` additionally expects
`windows-device-records.json`, which this scan does not create; it is not part of
the normal app setup. Device records and reports remain ignored by Git.

## Troubleshooting and checks

| Problem | What to check |
| --- | --- |
| `py` is not recognized | Install the Python launcher, reopen PowerShell, and verify `py -3.12 --version`. Alternatively use your Python 3.12 executable to create `.venv`. |
| `.venv\Scripts\python.exe` is missing | Run the environment-creation step from the repository root. ZIP downloads do not include an environment. |
| `ModuleNotFoundError` | Install the base dependencies and the relevant optional group using `.venv`'s Python, then restart. |
| llama.cpp compilation or wheel error | Use 64-bit Python 3.12 and the CPU wheel command above; consult the upstream installation guide if that wheel is unavailable. |
| Native DLL load failure | Check Python/package architecture and install the Microsoft Visual C++ runtime required by the failing package. Avoid mixing OpenCV variants. |
| Config file not found | Copy the example to `config/config.json` using step 4. |
| Hugging Face download fails | Check connectivity, free disk space, and access to the configured model repository; retry after fixing the cause. A compatible local GGUF can be supplied with `--model`. |
| Face/layer hash mismatch | Remove only the specific damaged model file named in the error and retry its download; do not change the expected hash to bypass verification. |
| Slow responses or out of memory | Close other large apps, reduce `--ctx`, and avoid running several model-heavy tools together. CPU vision/transcription can be slow. |
| YouTube extraction fails | Check `node --version`, update `yt-dlp[default]` in `.venv`, and consult yt-dlp's current site/runtime guidance. |
| Garbled terminal display | Use Windows Terminal or start with `--no-spinner`. |

Quick checks that do not download or run models:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe src\chat.py --help
.\.venv\Scripts\python.exe tests\test_calc.py
```

With the document packages installed, run
`.\.venv\Scripts\python.exe tests\test_reader.py`.
The real face-model smoke test requires `FACES_SAMPLE` and can download weights;
it is not required for setup. `scripts/audit.py` is a developer utility that uses
web access and the local model; it is also not a setup step.

## Local data and updates

Settings live in `config/config.json`. Chats are saved under `data/chats/`,
feature downloads and caches under `data/downloads/`, and debug logs under
`data/logs/` when logging is enabled. The public example disables debug logging.
Generated directories are created as needed; do not expect them in a fresh ZIP.

To update a Git clone, stop the app and run `git pull --ff-only`, then review any
changed setup instructions. Keep your local config; do not overwrite it with the
example. Compare new example settings when features add configuration keys.
For ZIP updates, preserve your local config, data, and model folders separately.

Git uses an explicit file allowlist. Local settings, keys, saved/loaded chat
context, logs, documents, downloads, weights, environments, and generated reports
stay local. See [the publishing inventory](docs/PUBLISHING.md) before committing.

## Repository layout

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
