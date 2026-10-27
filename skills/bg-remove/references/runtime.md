# Runtime setup

Use Python 3.12. From the repository checkout:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r skills/bg-remove/scripts/requirements.txt
./bg-remove input.jpg output.png
```

For a standalone installed skill, create an environment in a user-selected writable directory and install that skill's `scripts/requirements.txt`. Invoke its interpreter and the resolved `scripts/cutout.py` directly. Windows users can use the Python script directly; the convenience shell launcher is for POSIX shells.

The first segmentation run downloads the BiRefNet_HR model and Python model code from Hugging Face at revision `a7a562f6fd16021180f2f4348f4de003a2d3d1e1`. This is a substantial local download and executes the pinned model code (`trust_remote_code=True`). Explain that setup before a first download when the user has not authorized dependency/model installation. Images are processed locally; no inference API or image upload is used.

Cache follows an existing `HF_HOME`, otherwise `$XDG_CACHE_HOME/compound-bg-remove-public/hf` or `~/.cache/compound-bg-remove-public/hf`. Models and environments live outside the skill. No model weights ship in the repository. Once cached, `--offline` prevents the model loader from fetching files.

`auto` uses Apple MPS when available, otherwise CPU. Device allocation errors are reported rather than silently changing behavior. Retry once with `--device cpu` when an accelerator failure warrants it. CPU can be much slower. CUDA is not exposed in this release. The dependency versions were validated on Python 3.12/macOS arm64; Linux, Windows, and Intel Mac inference have not been verified.

Pinned direct dependencies improve repeatability but are not a complete platform-specific lockfile. A dependency resolver may reject this exact set on unsupported Python/platform combinations. Do not substitute package versions silently and claim the validated configuration.
