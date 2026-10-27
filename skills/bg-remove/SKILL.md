---
name: bg-remove
description: Remove a still image's background into a real transparent PNG using local segmentation and foreground edge decontamination. Use for product cutouts, isolated objects, or colored halos around cutout edges; not video matting.
---

# Background removal

Use the bundled `scripts/cutout.py` from this skill's actual installed directory. The same workflow works in Codex and Claude Code; it requires a shell and a Python environment with `scripts/requirements.txt` installed. Do not assume a fixed home directory or a sibling skill exists.

## Run

Resolve the input and requested output paths. Use the user's Python environment, or prepare a project-local Python 3.12 virtual environment when installing dependencies is authorized. Do not install packages into the system Python. See [runtime setup](references/runtime.md) when the environment or model cache is missing.

```sh
python /path/to/installed/bg-remove/scripts/cutout.py input.jpg output.png
python /path/to/installed/bg-remove/scripts/cutout.py --out cutouts input.jpg other.webp
python /path/to/installed/bg-remove/scripts/cutout.py --out cutouts --resume input.jpg other.webp
```

Replace the example interpreter and skill path with resolved paths. Outputs require `.png`; existing outputs are preserved unless `--overwrite` is requested. Batch resume skips only intact outputs whose input, configuration, and output hashes match its receipt. After a native runtime crash, rerun the same batch with `--resume`; do not repeatedly retry an unchanged failure.

An existing transparent image is preserved without remasking. Use `--remask` only when the user wants its existing cutout changed; existing holes still remain transparent. `--device cpu` is the fallback for an MPS failure. `--offline` requires previously cached model files.

## Preserve the useful edge behavior

The implementation predicts alpha with BiRefNet_HR at 2048 square, then re-estimates foreground RGB against a two-pixel-eroded copy of that mask. It exports the original alpha, not the eroded copy. This targets old-background color trapped inside otherwise opaque edge pixels without shrinking the final silhouette. Do not replace this with color-distance keying or silently erode the exported alpha.

## Verify the actual result

Check that the output decodes as RGBA PNG at the orientation-corrected input dimensions. Inspect the silhouette at native resolution on a dark, light, and strongly contrasting temporary ground. Check genuine holes, thin parts, and any pre-existing transparency. Segmentation is an estimate; report visible failures rather than promising perfect glass, hair, or shadow handling. The model commonly removes cast shadows; creating replacement shadows is a separate task.

Keep source images intact. Do not overwrite the source or add generated output to version control unless the user asks to retain it. Clean temporary composites after review. Report the output location, any failed batch inputs, and material limitations.
