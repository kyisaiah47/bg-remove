"""Regression tests use only generated temporary fixtures; no shipped media."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "skills/bg-remove/scripts/cutout.py"
spec = importlib.util.spec_from_file_location("cutout", SCRIPT)
cut = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cut)


class CutoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="bg-remove-test-", dir="/private/tmp" if Path("/private/tmp").is_dir() else None)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def image(self, name="input.png", transparent=False):
        from PIL import Image
        p = self.root / name
        im = Image.new("RGBA", (40, 32), (130, 90, 35, 255))
        if transparent:
            im.putpixel((0, 0), (1, 2, 3, 0))
            im.putpixel((1, 0), (4, 5, 6, 128))
        im.save(p)
        return p

    def test_alpha_bytes_unchanged_and_solve_mask_eroded(self):
        import numpy as np
        from PIL import Image
        alpha = np.zeros((32, 40), dtype=np.uint8)
        alpha[4:28, 5:35] = 255
        alpha[4, 5:35] = 128
        captured = []
        def solve(rgb, a, return_background):
            captured.append(a.copy())
            return rgb
        output = cut.decontaminate(Image.new("RGB", (40, 32), "red"), alpha, solver=solve)
        np.testing.assert_array_equal(np.asarray(output)[:, :, 3], alpha)
        self.assertLess(captured[0].sum(), (alpha.astype(float) / 255).sum())
        self.assertEqual(captured[0][16, 20], 1)
        self.assertEqual(captured[0][0, 0], 0)

    def test_existing_transparency_preserved_without_model(self):
        import numpy as np
        from PIL import Image
        src = self.image(transparent=True)
        with patch.object(cut, "load_model", side_effect=AssertionError("unexpected inference")):
            actual = cut.cutout(src)
        with Image.open(src) as expected:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_resume_requires_matching_source_and_intact_output(self):
        from PIL import Image
        src = self.image()
        out = self.root / "out"
        fake = Image.new("RGBA", (40, 32), (20, 30, 40, 128))
        with patch.object(cut, "cutout", return_value=fake) as run:
            self.assertEqual(cut.main(["--out", str(out), str(src)]), 0)
            self.assertEqual(cut.main(["--out", str(out), "--resume", str(src)]), 0)
            self.assertEqual(run.call_count, 1)
            target = out / "input.png"
            target.write_bytes(b"incomplete")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cut.main(["--out", str(out), "--resume", str(src)]), 1)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(cut.main(["--out", str(out), "--overwrite", str(src)]), 0)
            self.assertEqual(run.call_count, 2)
            src.write_bytes(b"changed")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cut.main(["--out", str(out), "--resume", str(src)]), 1)

    def test_batch_keeps_independent_success_and_reports_failure(self):
        from PIL import Image
        a, b = self.image("a.png"), self.image("b.png")
        out = self.root / "out"
        def infer(p, *_):
            if p == a:
                raise ValueError("bad input")
            return Image.new("RGBA", (40, 32), (20, 30, 40, 128))
        with patch.object(cut, "cutout", side_effect=infer), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cut.main(["--out", str(out), str(a), str(b)]), 1)
        self.assertFalse((out / "a.png").exists())
        self.assertTrue(cut.valid_rgba(out / "b.png"))

    def test_collision_and_source_overwrite_rejected_before_inference(self):
        a = self.image("same.png")
        sub = self.root / "sub"
        sub.mkdir()
        b = sub / "same.jpg"
        from PIL import Image
        Image.new("RGB", (8, 8)).save(b)
        for argv in [[str(a), str(a)], ["--out", str(self.root / "out"), str(a), str(b)]]:
            with self.assertRaises(SystemExit) as exc, contextlib.redirect_stderr(io.StringIO()):
                cut.main(argv)
            self.assertEqual(exc.exception.code, 2)

    def test_atomic_png_leaves_no_partial_output_on_error(self):
        class BrokenImage:
            def save(self, path, format):
                Path(path).write_bytes(b"partial")
                raise ValueError("simulated encoder failure")
        output = self.root / "out.png"
        with self.assertRaises(ValueError):
            cut.atomic_png(BrokenImage(), output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_foreground_color_is_recovered_on_synthetic_blend(self):
        import numpy as np
        import cv2
        from PIL import Image
        mask = np.zeros((64, 64), np.float32)
        mask[12:52, 12:52] = 1
        alpha = cv2.GaussianBlur(mask, (7, 7), 1.2)
        foreground = np.array([0.85, 0.12, 0.08])
        background = np.array([0.1, 0.85, 0.2])
        rgb = foreground[None, None, :] * alpha[:, :, None] + background[None, None, :] * (1 - alpha[:, :, None])
        a = np.rint(alpha * 255).astype(np.uint8)
        image = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
        actual = np.asarray(cut.decontaminate(image, a))
        band = (a > 50) & (a < 220)
        before = np.abs(np.asarray(image)[band].astype(float) / 255 - foreground).mean()
        after = np.abs(actual[:, :, :3][band].astype(float) / 255 - foreground).mean()
        self.assertLess(after, before)
        np.testing.assert_array_equal(actual[:, :, 3], a)

    @unittest.skipUnless(os.environ.get("BG_REMOVE_RUN_INFERENCE") == "1", "opt-in cached model smoke test")
    def test_cached_real_inference_smoke(self):
        from PIL import Image, ImageDraw
        src = self.root / "opaque-product.png"
        image = Image.new("RGB", (128, 96), (230, 230, 220))
        ImageDraw.Draw(image).rounded_rectangle((40, 15, 90, 82), radius=8, fill=(200, 50, 20))
        image.save(src)
        output = cut.cutout(src, device="mps", offline=True)
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(output.size, (128, 96))
        target = self.root / "real.png"
        cut.atomic_png(output, target)
        self.assertTrue(cut.valid_rgba(target))


if __name__ == "__main__":
    unittest.main()
