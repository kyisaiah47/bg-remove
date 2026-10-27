# bg-remove

A background-removal skill for **Codex and Claude Code**, with a local CLI. Produces RGBA PNG cutouts and corrects old-background color along the edge. No inference API, API key, or image upload.

A single shared skill lives in [`skills/bg-remove`](skills/bg-remove). Both platforms use the same instructions and Python implementation.

## Install

### Claude Code

```text
/plugin marketplace add kyisaiah47/bg-remove
/plugin install bg-remove@compound-bg-remove
```

Invoke `/bg-remove:bg-remove` or ask to remove a still image's background. Plugin installation does not install Python dependencies or model weights.

### Codex

Use the plugin listing when available, or install the standalone skill folder:

```sh
git clone https://github.com/kyisaiah47/bg-remove.git
mkdir -p ~/.agents/skills
cp -R bg-remove/skills/bg-remove ~/.agents/skills/bg-remove
```

If that destination already exists, compare it first; do not overwrite an existing skill silently. Start a new session and invoke `$bg-remove`.

### CLI and Python environment

From this checkout, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r skills/bg-remove/scripts/requirements.txt
./bg-remove input.jpg output.png
./bg-remove --out cutouts input.jpg other.webp
./bg-remove --out cutouts --resume input.jpg other.webp
```

Or set `BG_REMOVE_PYTHON` to an existing compatible interpreter. The launcher never installs dependencies automatically. In an installed plugin, invoke its resolved `scripts/cutout.py` with the prepared interpreter. Windows users can invoke that Python script directly.

## What changes at the edge

The BiRefNet_HR model estimates alpha at 2048×2048. PyMatting then estimates foreground RGB using a two-pixel-eroded copy of the alpha. The exported alpha remains unchanged. The erosion steers color cleanup; it does not shrink the output silhouette.

Example requests:

- “Cut this product out onto a transparent background; check the fine edges on dark and light.”
- “Batch these photos into `cutouts/`, preserving the sources.”
- “Resume the interrupted batch and report which images failed.”

Existing transparent inputs keep their current RGBA pixels by default. `--remask` infers a new mask while preserving existing transparent holes. Existing output files require `--overwrite`. Batch `--resume` uses input/configuration/output hashes and validates the PNG before skipping it. A corrupted, changed, or unrelated output is never counted as complete.

## Requirements and limits

- Python 3.12; pinned direct dependencies in `skills/bg-remove/scripts/requirements.txt`.
- First segmentation downloads a substantial model checkpoint and pinned Python model code from Hugging Face. It executes that model code through `trust_remote_code=True`; inspect the cited revision if your environment requires code review. No model code or weights are bundled.
- `--device auto` chooses Apple MPS when available, otherwise CPU. `--device cpu` is an explicit fallback. CUDA is not exposed.
- `--offline` uses an existing cache. Respect `HF_HOME`, or cache under `$XDG_CACHE_HOME/compound-bg-remove-public/hf` (default `~/.cache/compound-bg-remove-public/hf`).
- Still images only. Glass, hair, tiny structures, and cast shadows can need manual work. Removing a cast shadow is not proof that the foreground color is correct.
- EXIF orientation is applied. Color profiles and ancillary metadata are not preserved. Very large inputs consume substantial memory, particularly during foreground estimation.
- Shell-free `--help` and argument validation work without model dependencies. Runtime tests cover macOS arm64/Python 3.12; other platforms are unverified.

This publication copy is separate from the author's installed workflow. It does not install hooks, alter other skills, or register background services.

## Verify or reproduce

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The tests create temporary synthetic fixtures, verify alpha/edge behavior and output handling, then delete them. Set `BG_REMOVE_RUN_INFERENCE=1` and provide a cached model via `HF_HOME` to enable the slower real-inference smoke test. The test reports completion and dimensions, not segmentation quality on natural photographs. Tests create no reusable image cache.

## Distribution and licensing

Copyright 2026 Compound Labs. All rights reserved. The repository is publicly distributed under the limited usage permission in [`LICENSE`](LICENSE). Third-party dependencies and the downloaded model keep their own licenses. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Packaging references: [Claude Code plugins](https://code.claude.com/docs/en/plugins-reference), [OpenAI portable plugin packaging](https://developers.openai.com/plugins/build/plugins).
