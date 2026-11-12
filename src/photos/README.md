# photos/

Local photo tools that run ONNX/OpenCV models on the CPU: face recognition (`/faces`) and background
removal / depth layers (`/cutout`, `/layers`). Each tool has the same three layers: a command module,
pure maths, and model loading.

| Layer | `/faces` | `/cutout` + `/layers` |
| --- | --- | --- |
| Command and flow | `faces.py` | `layers.py` |
| Pure maths (numpy only, unit-testable) | `face_match.py` | `layer_ops.py` |
| Models: download, checksum, ONNX sessions | `face_models.py` | `layer_models.py` |
| Storage / library on disk | `face_store.py` (SQLite DB), `face_sync.py` (people folders) | — |
| Chat follow-ups | `faces_followup.py` | — |

**Why they're together:** both tools follow the same pattern, sharing model download with checksums,
CPU inference, image loading, and pure maths kept separate so it's testable without models. They
also call each other: `layers.py` uses `faces.find_path` to resolve the photo the user means.
`mcp_server.py` (at `src/`) serves the `/layers` engine to Claude Code / Claude Desktop.

The file prefixes (`face_*`, `layer_*`) still show which tool a file belongs to. If this folder keeps
growing, split it into `photos/faces/` and `photos/layers/`.

`faces.py` ↔ `faces_followup.py` is a circular pair. `faces.py` imports the follow-up module
**lazily**, inside a function. Keep it that way.

Self-checks on a real photo (run from `src/`):

```
..\.venv\Scripts\python.exe -m photos.face_models <photo>
..\.venv\Scripts\python.exe -m photos.layer_models --selfcheck <photo>
```
