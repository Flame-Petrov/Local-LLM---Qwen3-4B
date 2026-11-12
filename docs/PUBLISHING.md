# Public repository inventory

This is the reviewed publishing set. `.gitignore` ignores all files by default
and allows only the exact paths below. New files, even Python or Markdown files
inside source folders, must be reviewed and explicitly allowed before publishing.

## Included

Application source, offline test source, utility source, package documentation,
the launcher, and sanitized example configuration. The price dictionary example
contains reusable parsing rules; the original audit-derived local dictionary and
its audit output are excluded. Test contact details use reserved example domains
and fictional phone numbers. Generic fixture paths and public service/model
identifiers are retained.

Exact public file list (59 files):

- `.gitignore`
- `README.md`
- `bluetooth-scan/report.py`
- `bluetooth-scan/scan.py`
- `config/config.example.json`
- `config/page_dict.example.json`
- `docs/PUBLISHING.md`
- `run.bat`
- `scripts/audit.py`
- `src/chat.py`
- `src/core/README.md`
- `src/core/__init__.py`
- `src/core/calc.py`
- `src/core/config.py`
- `src/core/debuglog.py`
- `src/core/live.py`
- `src/core/ui.py`
- `src/files/README.md`
- `src/files/__init__.py`
- `src/files/reader.py`
- `src/files/writer.py`
- `src/mcp_server.py`
- `src/media/README.md`
- `src/media/__init__.py`
- `src/media/music.py`
- `src/media/music_followup.py`
- `src/media/see.py`
- `src/media/video.py`
- `src/photos/README.md`
- `src/photos/__init__.py`
- `src/photos/face_match.py`
- `src/photos/face_models.py`
- `src/photos/face_store.py`
- `src/photos/face_sync.py`
- `src/photos/faces.py`
- `src/photos/faces_followup.py`
- `src/photos/layer_models.py`
- `src/photos/layer_ops.py`
- `src/photos/layers.py`
- `src/web/README.md`
- `src/web/__init__.py`
- `src/web/agent.py`
- `src/web/images.py`
- `src/web/planner.py`
- `src/web/prices.py`
- `src/web/profiles.py`
- `src/web/search.py`
- `tests/layers_fakes.py`
- `tests/run_mcp_with_fakes.py`
- `tests/test_calc.py`
- `tests/test_face_models.py`
- `tests/test_faces.py`
- `tests/test_faces_followup.py`
- `tests/test_layers.py`
- `tests/test_mcp_server.py`
- `tests/test_music.py`
- `tests/test_music_followup.py`
- `tests/test_reader.py`
- `tests/test_writer.py`

## Excluded; preserved on disk

- `config/config.json` and `config/page_dict.json`: local settings and original local dictionary.
- `data/`: saved and loaded chats, memory/context, logs, read caches, documents, downloads, audit output, face databases, embeddings, feedback, and other runtime state.
- `models/`, `bin/`, `downloads/`: model weights, binaries, and downloaded assets.
- `.venv/`, `.bluetooth-scan-deps/`, Python caches, build output, and dependency caches.
- `plans/`: development notes, review history, and task context.
- `docs/file-inventory.txt` and `docs/debug-images/`: historical inventory and debug screenshots.
- Bluetooth JSON/TXT reports: nearby-device names, addresses, and stored device records.
- Environment files, credentials, keys, cookies, editor settings, agent context,
  backups, archives, and all other paths not explicitly listed above.

No private files were deleted. Git metadata is not part of the publishing set.
No existing commits or configured remotes were present during this review.

## Before committing and pushing

From the repository root, inspect the exact proposed staging set:

```powershell
git status --short --untracked-files=all
git add --dry-run .
```

Only the listed files should appear. Stage normally after reviewing:

```powershell
git add .
git diff --cached --stat
git diff --cached --check
git diff --cached
```

Inspect the staged contents before making a commit. `.gitignore` does not stop
forced additions or protect files already tracked; avoid `git add -f`. Review
future edits to public source and example configuration for private data too.
Set a Git author identity suitable for publication, such as your GitHub noreply
address, before committing: author names and emails are public commit metadata.
Then create your initial commit, choose the branch name, configure your intended
GitHub remote, and push. This preparation does not commit, change branches,
configure a remote, or push.

## Validation limits

The publishing review checks the proposed file set and common credential/personal
identifier patterns. It is not a guarantee that every possible secret format can
be detected. Downloaded assets and private runtime data are excluded as whole
directories and do not need to be opened for publication.

A clean-machine dependency install and real model/network execution are not
validated by this preparation. There is no dependency lockfile or license file;
no license has been selected on the author's behalf.
