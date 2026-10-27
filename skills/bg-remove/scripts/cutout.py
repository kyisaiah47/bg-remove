#!/usr/bin/env python3
"""Local image cutouts with preserved alpha and foreground edge decontamination."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

MODEL = "ZhengPeng7/BiRefNet_HR"
REVISION = "a7a562f6fd16021180f2f4348f4de003a2d3d1e1"
SIZE = 2048
VERSION = "0.1.2"
_MODEL = None


def dependencies():
    try:
        import numpy as np
        import cv2
        from PIL import Image, ImageOps
        from pymatting import estimate_foreground_ml
        return np, cv2, Image, ImageOps, estimate_foreground_ml
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Install scripts/requirements.txt in your Python 3.12 environment: " + str(exc)) from exc


def load_model(device: str, offline: bool):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    from transformers import AutoModelForImageSegmentation
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; choose --device cpu")
    # The model repository includes Python code. Pin both code and weights to one fixed revision.
    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL, revision=REVISION, code_revision=REVISION,
        trust_remote_code=True, local_files_only=offline,
    ).to(torch.float32).to(device).eval()
    _MODEL = model, device
    return _MODEL


def decontaminate(image, alpha, erode=2, solver=None):
    """Solve foreground RGB using an eroded mask; return the ORIGINAL alpha bytes."""
    np, cv2, Image, _, default_solver = dependencies()
    if alpha.shape != (image.height, image.width):
        raise ValueError("alpha dimensions differ from image")
    if alpha.dtype != np.uint8:
        raise ValueError("alpha must contain uint8 coverage values")
    a = alpha.astype(np.float64) / 255.0
    if not np.any(a):
        return Image.fromarray(np.zeros((image.height, image.width, 4), dtype=np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode * 2 + 1,) * 2)
    solve_alpha = cv2.erode(a, k) if erode else a
    foreground = (solver or default_solver)(
        np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0,
        solve_alpha, return_background=False,
    )
    rgb = np.clip(foreground * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def cutout(path: Path, device="auto", offline=False, remask=False):
    np, _, Image, ImageOps, _ = dependencies()
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("animated images are unsupported; supply one still frame")
        source = ImageOps.exif_transpose(source).convert("RGBA")
        # Preserve a pre-existing cutout unless explicitly asked to infer a new mask.
        if not remask and source.getextrema()[3][0] < 255:
            return source.copy()
        image = source.convert("RGB")
    import torch
    from torchvision import transforms as T
    model, selected = load_model(device, offline)
    transform = T.Compose([
        T.Resize((SIZE, SIZE)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.inference_mode():
        pred = model(transform(image).unsqueeze(0).to(selected))[-1].sigmoid().float().cpu()[0].squeeze().numpy()
    mask = Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8)).resize(image.size, Image.Resampling.BICUBIC)
    alpha = np.asarray(mask, dtype=np.uint8)
    # Explicit remasking may remove more background but must not fill existing transparent holes.
    if remask:
        original = np.asarray(source.getchannel("A"), dtype=np.float64) / 255.0
        alpha = np.rint(alpha.astype(np.float64) * original).astype(np.uint8)
    return decontaminate(image, alpha)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix=".bg-remove-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def valid_rgba(path):
    try:
        _, _, Image, _, _ = dependencies()
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGBA":
                return False
            image.load()
        return True
    except (OSError, ValueError):
        return False


def atomic_png(image, destination):
    fd, name = tempfile.mkstemp(prefix=".bg-remove-", suffix=".png", dir=destination.parent)
    os.close(fd)
    try:
        image.save(name, format="PNG")
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path, help="INPUT OUTPUT.png, or INPUT... with --out")
    p.add_argument("--out", type=Path, help="batch destination directory")
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto", help="auto selects MPS when available, otherwise CPU")
    p.add_argument("--offline", action="store_true", help="use only cached model files")
    p.add_argument("--overwrite", action="store_true", help="replace existing output files")
    p.add_argument("--resume", action="store_true", help="batch only: skip verified outputs matching the saved input/config hashes")
    p.add_argument("--remask", action="store_true", help="infer a new mask even when the source already has transparency")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.resume and not args.out:
        p.error("--resume requires --out")
    if args.overwrite and args.resume:
        p.error("choose --overwrite or --resume, not both")
    if not args.out and len(args.paths) != 2:
        p.error("supply INPUT OUTPUT.png, or --out DIRECTORY INPUT...")
    pairs = [(src, args.out / (src.stem + ".png")) for src in args.paths] if args.out else [tuple(args.paths)]
    outputs = [dst.resolve() for _, dst in pairs]
    if len(set(outputs)) != len(outputs):
        p.error("input basenames collide in the output directory")
    for src, dst in pairs:
        if not src.is_file():
            p.error(f"input is not a file: {src}")
        if dst.suffix.lower() != ".png":
            p.error(f"output must end in .png: {dst}")
        if src.resolve() == dst.resolve():
            p.error("input and output must differ")
        if dst.exists() and not (args.resume or args.overwrite):
            p.error(f"output already exists; choose --overwrite or batch --resume: {dst}")
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "compound-bg-remove-public" / "hf"
    os.environ.setdefault("HF_HOME", str(cache))
    config = {"version": VERSION, "model": MODEL, "revision": REVISION, "device": args.device, "remask": args.remask}
    state_path = args.out / ".bg-remove-state.json" if args.out else None
    state = {}
    if args.resume and state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            if not isinstance(state, dict):
                raise ValueError("state must be an object")
        except (OSError, ValueError) as exc:
            p.error(f"cannot read resume state: {exc}")
    failures = 0
    for src, dst in pairs:
        try:
            source_hash = digest(src)
            key = str(dst.resolve())
            old = state.get(key, {})
            if dst.exists() and args.resume:
                if (old.get("source") == source_hash and old.get("config") == config
                        and old.get("output") == digest(dst) and valid_rgba(dst)):
                    print(f"skipped verified output: {dst}")
                    continue
                raise ValueError(f"existing output does not match resume state: {dst}; remove it or choose --overwrite")
            dst.parent.mkdir(parents=True, exist_ok=True)
            result = cutout(src, args.device, args.offline, args.remask)
            atomic_png(result, dst)
            state[key] = {"source": source_hash, "config": config, "output": digest(dst)}
            if state_path:
                atomic_json(state_path, state)
            print(f"wrote {dst}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"bg-remove: {src}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
