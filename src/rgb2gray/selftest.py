"""
selftest.py -- correctness checks for the conversion engine.

    python selftest.py            # exit 0 = everything passed
    python selftest.py --keep     # leave the fixture behind for inspection

Not a benchmark.  It builds its own small fixture -- mixed formats, a PNG
deliberately named ``.bmp``, two sources fighting for one output name, an empty
directory, an unpaired ``.json``, a non-image file -- runs the engine over it in
every output format, and asserts the invariants the app promises.  Each check
prints its own name, so the run is its own list of what is guaranteed.

This is the project's only automated test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from . import gray_core as gc


RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    RESULTS.append((bool(ok), name, detail))


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def annotation(image_name: str) -> str:
    """A LabelMe / X-AnyLabeling shaped annotation, formatted like the real ones."""
    return json.dumps({
        "version": "4.0.2",
        "flags": {},
        "shapes": [{"label": "STICKER", "points": [[10.5, 20.25], [30.0, 40.0]],
                    "shape_type": "polygon", "flags": {}}],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": 96,
        "imageWidth": 128,
    }, indent=2, ensure_ascii=False)


def build_fixture(root: Path) -> None:
    nested = root / "a" / "deep nest"
    nested.mkdir(parents=True)
    (root / "b").mkdir()
    (root / "empty_dir").mkdir()          # must still be mirrored

    rng = np.random.default_rng(0)
    rgb = Image.fromarray(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8), "RGB")

    for name, folder in (("one.bmp", nested), ("two.bmp", root / "b")):
        rgb.save(folder / name)
        (folder / name).with_suffix(".json").write_text(annotation(name))

    rgb.save(root / "b" / "plain.png")
    rgb.save(root / "b" / "plain.jpg", quality=92)
    rgb.save(root / "a" / "liar.bmp", format="PNG")      # PNG wearing a .bmp name
    (root / "b" / "notes.txt").write_text("not an image")
    (root / "b" / "orphan.json").write_text(annotation("nothing.bmp"))


# one.bmp two.bmp plain.jpg plain.png liar.bmp -- but plain.jpg and plain.png
# share a stem, so with a single output format one of them is shadowed and the
# run should convert four, not five, and say so.
EXPECTED_SOURCES = 5
EXPECTED_IMAGES = 4
EXPECTED_SHADOWED = 1
EXPECTED_DIRS = {"a", "a/deep nest", "b", "empty_dir"}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def run(src: Path, dst: Path, fmt: str, **kw) -> gc.RunStats:
    return gc.convert_dataset(src, dst, mode=kw.pop("mode", "cpu"),
                              out_format=fmt, workers=kw.pop("workers", 2), **kw)


def output_images(out: Path) -> list[Path]:
    return [p for p in out.rglob("*") if p.is_file()
            and p.suffix.lower() not in (".json", ".csv")]


def check_structure(src: Path, out: Path, st: gc.RunStats, fmt: str) -> None:
    suffix = gc.OUTPUT_FORMATS[fmt].suffix
    dirs = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_dir()}
    check(dirs == EXPECTED_DIRS, f"[{fmt}] tree mirrored incl. empty dirs",
          f"got {sorted(dirs)}")
    check(st.ok == EXPECTED_IMAGES, f"[{fmt}] all {EXPECTED_IMAGES} images converted",
          f"converted {st.ok}, failed {st.failed}")
    check(st.failed == 0, f"[{fmt}] no errors", str(st.errors))

    images = output_images(out)
    # The report must match the filesystem. Two sources sharing a stem used to
    # be counted as two conversions while producing one file.
    check(len(images) == st.ok,
          f"[{fmt}] reported count matches the files on disk",
          f"reported {st.ok}, on disk {len(images)}")
    check(all(p.suffix == suffix for p in images),
          f"[{fmt}] every output carries the chosen extension")
    modes = {Image.open(p).mode for p in images}
    check(modes == {"L"}, f"[{fmt}] every output is 8-bit single channel",
          f"modes {modes}")

    # liar.bmp wears a .bmp name but holds PNG bytes, and plain.png is the
    # shadowed one -- so a PNG entry here can only have come from the content.
    check(st.formats_seen == {"BMP": 2, "PNG": 1, "JPEG": 1},
          f"[{fmt}] format taken from content, not from the extension",
          f"formats seen {st.formats_seen}")
    check(not (out / "b" / "orphan.json").exists(),
          f"[{fmt}] unpaired .json not copied")
    check(not (out / "b" / "notes.txt").exists(),
          f"[{fmt}] non-image file not copied")
    check(st.ann_copied == 2, f"[{fmt}] both paired .json copied",
          f"copied {st.ann_copied}")


def check_shadowing(src: Path, tmp: Path) -> None:
    """A same-stem collision must be resolved once, deterministically, and said."""
    scan = gc.scan_input(src, tmp / "never_written")
    check(len(scan.shadowed) == EXPECTED_SHADOWED,
          "[shadow] the same-stem collision is detected and reported",
          f"shadowed {scan.shadowed}")
    check(len(scan.tasks) == EXPECTED_IMAGES,
          "[shadow] the shadowed source is not queued",
          f"queued {len(scan.tasks)} of {EXPECTED_SOURCES} sources")

    # Same input, repeated runs: the surviving image must always be the same one.
    digests = set()
    for i in range(4):
        out = tmp / f"det{i}"
        run(src, out, "bmp")
        digests.add(hashlib.md5((out / "b" / "plain.bmp").read_bytes()).hexdigest())
        shutil.rmtree(out)
    check(len(digests) == 1,
          "[shadow] which source wins is deterministic across runs",
          f"{len(digests)} different results from identical runs")


def check_annotation(src: Path, out: Path, fmt: str) -> None:
    rel = Path("a/deep nest/one.json")
    original = (src / rel).read_text()
    copied = (out / rel).read_text()
    suffix = gc.OUTPUT_FORMATS[fmt].suffix
    if suffix == ".bmp":
        check(copied == original, "[bmp] .json byte-identical when extension is kept")
        return
    a, b = json.loads(original), json.loads(copied)
    check(b["imagePath"] == f"one{suffix}",
          f"[{fmt}] imagePath rewritten to the new extension", b["imagePath"])
    check(all(a[k] == b[k] for k in a if k != "imagePath"),
          f"[{fmt}] every other .json field untouched")
    check(a["shapes"] == b["shapes"], f"[{fmt}] annotation shapes preserved")
    changed = [l for l in copied.splitlines() if l not in original.splitlines()]
    check(len(changed) == 1, f"[{fmt}] exactly one line of the .json changed",
          f"{len(changed)} lines differ")


def check_lossless(outs: dict[str, Path]) -> None:
    ref = np.asarray(Image.open(outs["bmp"] / "a/deep nest/one.bmp"), dtype=np.int16)
    for fmt, lossless in (("png", True), ("tiff", True), ("jpeg", False)):
        p = outs[fmt] / f"a/deep nest/one{gc.OUTPUT_FORMATS[fmt].suffix}"
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.int16)
        diff = int(np.abs(arr - ref).max())
        if lossless:
            check(diff == 0, f"[{fmt}] output is lossless vs BMP", f"max diff {diff}")
        else:
            check(diff > 0, "[jpeg] output is measurably lossy, as labelled",
                  f"max diff {diff}")


def check_gpu(src: Path, tmp: Path, cpu_out: Path) -> None:
    if gc.torch_module() is None:
        check(True, "[gpu] skipped - PyTorch not installed", "skipped")
        return
    device, reason = gc.detect_gpu()
    out = tmp / "gpu"
    try:
        run(src, out, "bmp", mode="gpu", device=device or "cpu", batch_size=2)
    except Exception as exc:
        check(False, "[gpu] pipeline ran", f"{type(exc).__name__}: {exc}")
        return
    worst = 0
    for p in cpu_out.rglob("*.bmp"):
        q = out / p.relative_to(cpu_out)
        if not q.exists():
            check(False, "[gpu] produced the same file set", f"missing {q.name}")
            return
        a = np.asarray(Image.open(p), dtype=np.int16)
        b = np.asarray(Image.open(q), dtype=np.int16)
        worst = max(worst, int(np.abs(a - b).max()))
    label = "cuda" if device else "cpu-fallback"
    check(worst <= 1, f"[gpu:{label}] agrees with CPU mode within 1 LSB",
          f"max diff {worst}")

    # workers=1 is a separate implementation, not just fewer threads, so it
    # needs its own proof that it produces the same bytes.
    serial_out = tmp / "gpu_serial"
    st = run(src, serial_out, "bmp", mode="gpu", device=device or "cpu",
             batch_size=2, workers=1)
    check(st.workers == 1, "[gpu:serial] reports a single worker",
          f"reported {st.workers}")
    check(st.ok == EXPECTED_IMAGES, "[gpu:serial] converted every image",
          f"{st.ok} of {EXPECTED_IMAGES}")
    same = all(
        (serial_out / p.relative_to(out)).exists()
        and hashlib.md5((serial_out / p.relative_to(out)).read_bytes()).hexdigest()
        == hashlib.md5(p.read_bytes()).hexdigest()
        for p in out.rglob("*.bmp"))
    check(same, "[gpu:serial] byte-identical to the threaded pipeline")


def check_levels(src: Path, tmp: Path) -> None:
    """The compression control must change the bytes without changing the pixels.

    Pinned to Pillow so the ladder is walked where the control is live: under
    OpenCV's ``rle`` the level is deliberately inert, and a 0 there is lifted to
    1.  That behaviour is checked in :func:`check_encoders`.
    """
    for key, fmt in gc.OUTPUT_FORMATS.items():
        knob = fmt.level
        if knob is None:
            # A format with no control must ignore one rather than crash.
            out = tmp / f"lvl_{key}_ignored"
            st = run(src, out, key, level=7, encoder="pillow")
            check(st.failed == 0 and st.level is None,
                  f"[level:{key}] no control - a level is ignored, not applied",
                  f"failed={st.failed} level={st.level}")
            continue

        sizes, pixels = {}, {}
        for value in (knob.lo, knob.default, knob.hi):
            out = tmp / f"lvl_{key}_{value}"
            st = run(src, out, key, level=value, encoder="pillow")
            check(st.level == value, f"[level:{key}] run records level {value}",
                  f"recorded {st.level}")
            files = sorted(output_images(out))
            sizes[value] = sum(p.stat().st_size for p in files)
            pixels[value] = [np.asarray(Image.open(p), dtype=np.int16) for p in files]

        check(sizes[knob.hi] != sizes[knob.lo],
              f"[level:{key}] the level actually reaches the encoder",
              f"{knob.lo} and {knob.hi} produced the same {sizes[knob.lo]} bytes")

        if fmt.lossless:
            worst = max(int(np.abs(a - b).max())
                        for a, b in zip(pixels[knob.lo], pixels[knob.hi]))
            check(worst == 0,
                  f"[level:{key}] every level is lossless - identical pixels",
                  f"max diff {worst}")
            check(sizes[knob.hi] < sizes[knob.lo],
                  f"[level:{key}] a higher level is smaller",
                  f"{knob.lo}: {sizes[knob.lo]}  {knob.hi}: {sizes[knob.hi]}")
        else:
            check(sizes[knob.hi] > sizes[knob.lo],
                  f"[level:{key}] a higher quality is larger",
                  f"{knob.lo}: {sizes[knob.lo]}  {knob.hi}: {sizes[knob.hi]}")

    # Out-of-range input is clamped, never passed to Pillow as-is.
    check(gc.resolve_format("png", 99, png_strategy="default").effective_level == 9
          and gc.resolve_format("png", -5, png_strategy="default").effective_level == 0,
          "[level] out-of-range values are clamped to the control's range")
    # Settings that cannot apply must not be reported as if they had.
    leftovers = [(f, e) for f in ("bmp", "tiff", "jpeg") for e in gc.ENCODERS]
    leftovers += [("png", "pillow")]
    stale = [(f, e) for f, e in leftovers
             if gc.resolve_format(f, 1, encoder=e, png_strategy="rle").png_strategy]
    check(not stale, "[level] a strategy is dropped where it cannot apply", str(stale))


def check_encoders(src: Path, tmp: Path, pillow_outs: dict) -> None:
    """OpenCV is a second *encoder*, so it must produce the same picture.

    Not the same bytes -- the two libraries lay a PNG out differently, and under
    the ``rle`` strategy the deflate stream is structurally different -- so the
    check is on decoded pixels, like the CPU-vs-GPU one.
    """
    if gc.cv2_module() is None:
        check(True, "[encoder] skipped - OpenCV not installed", "skipped")
        return

    for fmt in ("bmp", "png", "tiff"):        # lossless ones only
        out = tmp / f"cv_{fmt}"
        st = run(src, out, fmt, encoder="opencv", png_strategy="rle")
        check(st.failed == 0 and st.encoder == "opencv",
              f"[encoder:{fmt}] OpenCV encoder ran and is recorded",
              f"failed={st.failed} encoder={st.encoder!r}")
        # A strategy left over from a PNG run must not be recorded against a
        # format that has no strategy at all.
        check(st.png_strategy == ("rle" if fmt == "png" else ""),
              f"[encoder:{fmt}] only PNG records a strategy",
              f"recorded {st.png_strategy!r}")
        worst = 0
        for a in output_images(pillow_outs[fmt]):
            b = out / a.relative_to(pillow_outs[fmt])
            if not b.exists():
                check(False, f"[encoder:{fmt}] same file set", f"missing {b.name}")
                return
            pa = np.asarray(Image.open(a).convert("L"), dtype=np.int16)
            pb = np.asarray(Image.open(b).convert("L"), dtype=np.int16)
            worst = max(worst, int(np.abs(pa - pb).max()))
        check(worst == 0, f"[encoder:{fmt}] pixels identical to Pillow's encoder",
              f"max diff {worst}")

    # The strategy must reach the encoder, and must be PNG+OpenCV only.
    sizes = {}
    for strategy in gc.PNG_STRATEGIES:
        out = tmp / f"cv_png_{strategy}"
        st = run(src, out, "png", encoder="opencv", png_strategy=strategy)
        check(st.png_strategy == strategy,
              f"[encoder] png_strategy {strategy} is recorded", f"got {st.png_strategy}")
        sizes[strategy] = sum(p.stat().st_size for p in output_images(out))
    check(len(set(sizes.values())) > 1,
          "[encoder] the PNG strategy actually reaches the encoder", str(sizes))

    # rle caps zlib's match distance at 1, so levels 1-9 have nothing left to
    # search -- but level 0 still stores, so it must be floored rather than
    # applied: the UI greys the control out under rle, and a stale 0 would
    # otherwise hand back files 4.6x larger than the greyed control implies.
    def rle_run(name, level):
        return run(src, tmp / name, "png", encoder="opencv",
                   png_strategy="rle", level=level)

    def digests(name):
        return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
                for p in output_images(tmp / name)}

    a, b, z = rle_run("rle_l1", 1), rle_run("rle_l9", 9), rle_run("rle_l0", 0)
    check(digests("rle_l1") == digests("rle_l9")
          and a.ok == b.ok == EXPECTED_IMAGES,
          "[encoder] under rle levels 1-9 are byte-identical")
    check(digests("rle_l0") == digests("rle_l1") and z.level == 1,
          "[encoder] under rle level 0 is floored to 1, not left to store",
          f"recorded level {z.level}")


def check_format_defaults(src: Path, tmp: Path) -> None:
    """One width and one encoder, whatever the format.

    The measured optimum does move with the format -- BMP flattens at the
    physical core count while the compressing formats keep climbing to the
    logical one -- but the default deliberately does not follow it: those
    figures are one machine's, and a control that moves itself when you pick a
    format is harder to reason about than one you set once.  The finding lives
    in the stepper's hint and in README instead.
    """
    check(gc.default_workers() == gc.physical_cores(),
          "[default] one width for every format: the physical core count",
          f"got {gc.default_workers()}, want {gc.physical_cores()}")
    check(gc.DEFAULT_ENCODER == "pillow",
          "[default] one encoder for every format: Pillow")

    st = gc.convert_dataset(src, tmp / "def_png", mode="cpu", out_format="png")
    check(st.encoder == "pillow" and st.failed == 0,
          "[default] a PNG run with no settings stays on Pillow",
          f"encoder={st.encoder!r}")
    # The strategy default only bites once OpenCV has been chosen deliberately.
    st = gc.convert_dataset(src, tmp / "def_png_cv", mode="cpu", out_format="png",
                            encoder="opencv")
    check(st.png_strategy == "rle" and st.failed == 0,
          "[default] choosing OpenCV for PNG lands on rle",
          f"strategy={st.png_strategy!r}")


def check_shared_default(src: Path, tmp: Path) -> None:
    """Both modes must default to the same width, and spend it differently.

    REPORT.md's comparison is only readable if the two modes differ in one
    thing.  A mode quietly picking its own number would break that silently.
    """
    want = min(gc.default_workers(), EXPECTED_IMAGES)  # run_cpu caps at the queue
    st = gc.convert_dataset(src, tmp / "def_cpu", mode="cpu",
                            out_format="bmp", workers=None)
    check(st.workers == want, "[default] CPU width is default_workers()",
          f"got {st.workers}, want {want}")
    check(st.threads == st.workers,
          "[default] CPU spends one process per unit of width",
          f"width {st.workers} cost {st.threads}")

    if gc.torch_module() is None:
        check(True, "[default] GPU mode skipped - PyTorch not installed", "skipped")
        return
    device, _reason = gc.detect_gpu()
    st_gpu = gc.convert_dataset(src, tmp / "def_gpu", mode="gpu",
                                device=device or "cpu",
                                out_format="bmp", workers=None)
    check(st_gpu.workers == want, "[default] GPU width is the same default_workers()",
          f"got {st_gpu.workers}, want {want}")
    check(st_gpu.threads == 2 * st_gpu.workers,
          "[default] GPU spends two threads per unit of width (decode + encode)",
          f"width {st_gpu.workers} cost {st_gpu.threads}")
    # The whole point of a shared unit -- and both modes must cap it at the
    # queue length, or a short run spawns workers with nothing to do.
    check(st.workers == st_gpu.workers,
          "[default] both modes report the same width",
          f"cpu {st.workers} vs gpu {st_gpu.workers}")


def check_cancel(src: Path, tmp: Path) -> None:
    out = tmp / "cancelled"
    cancel = threading.Event()
    cancel.set()                                   # abort before anything runs
    st = gc.convert_dataset(src, out, mode="cpu", out_format="bmp", workers=2,
                            cancel=cancel)
    leftovers = list(out.rglob(".*.part*"))
    check(not leftovers, "[cancel] no staging files left behind", str(leftovers))
    on_disk = [p for p in out.rglob("*") if p.is_file() and p.suffix == ".bmp"]
    for p in on_disk:                              # nothing half-written
        try:
            Image.open(p).load()
        except Exception as exc:
            check(False, "[cancel] every file on disk is complete", f"{p.name}: {exc}")
            return
    check(True, "[cancel] every file on disk is complete",
          f"{len(on_disk)} file(s), run reported cancelled={st.cancelled}")


def check_log(src: Path, tmp: Path, log: Path) -> None:
    # A log that cannot be written must be reported, not swallowed: the UI used
    # to print "logged <path>" for a file that was never created.
    st = run(src, tmp / "logfail", "bmp",
             log_path="/proc/definitely-not-writable/runs.csv")
    check(st.ok == EXPECTED_IMAGES and st.log_error is not None,
          "[log] an unwritable log is reported, and does not fail the run",
          f"ok={st.ok} log_error={st.log_error!r}")

    check(log.exists(), "[log] run log written")
    if not log.exists():
        return
    with log.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    check(bool(rows), "[log] run log has at least one row")
    if not rows:
        return
    # The writer drops any key that is not a declared column, so a row that
    # silently lost a field would otherwise never be noticed.
    def sample_record() -> dict:
        return gc.build_run_record(gc.RunStats(), log.parent, log.parent,
                                   gc.scan_input(log.parent, log.parent))

    produced = set(sample_record())
    check(produced == set(gc.RUN_LOG_COLUMNS),
          "[log] every produced field has a column",
          f"only in record: {sorted(produced - set(gc.RUN_LOG_COLUMNS))}; "
          f"only in columns: {sorted(set(gc.RUN_LOG_COLUMNS) - produced)}")
    row = rows[-1]
    check(list(row) == list(gc.RUN_LOG_COLUMNS),
          "[log] header matches the declared column order")
    check(int(row["physical_cores"] or 0) >= 1,
          "[log] machine columns record physical cores")
    check(int(row["converted"] or 0) == EXPECTED_IMAGES,
          "[log] converted count matches the run")

    # A second run must append a row, not a second header.
    before = len(rows)
    gc.append_run_log(log, sample_record())
    with log.open(newline="", encoding="utf-8") as fh:
        again = list(csv.DictReader(fh))
    check(len(again) == before + 1, "[log] appending adds one row, not a header")


def check_failure_log(tmp: Path) -> None:
    """A run that fails must say *which* files, not just how many."""
    src = tmp / "broken_input"
    (src / "sub").mkdir(parents=True)
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8),
                    "RGB").save(src / "good.bmp")
    # Scanning matches on suffix, so both are picked up as work and then fail at
    # read -- as a corrupt file in a real dataset would.  Two shapes on purpose:
    # a plausible BMP header that decodes to nothing raises OSError, while a
    # file Pillow cannot place at all is the UnidentifiedImageError path.  The
    # histogram is only worth a column if it can tell them apart.
    (src / "broken.bmp").write_bytes(b"BM not actually a bitmap")
    (src / "sub" / "empty.bmp").write_bytes(b"")

    log = tmp / "failure_case" / "runs.csv"
    st = run(src, tmp / "broken_out", "bmp", log_path=log)
    check(st.ok == 1 and st.failed == 2,
          "[failures] the run converts what it can and fails the rest",
          f"ok={st.ok} failed={st.failed}")
    check(st.error_kinds == {"OSError": 1, "unreadable": 1},
          "[failures] each failure is counted under its own kind",
          str(st.error_kinds))

    fail_log = gc.failure_log_path(log)
    check(fail_log.exists(), "[failures] failure log written", str(fail_log))
    if not fail_log.exists():
        return
    with fail_log.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    check(list(rows[0]) == list(gc.FAILURE_LOG_COLUMNS),
          "[failures] header matches the declared column order")
    check(len(rows) == 2, "[failures] one row per failed file", f"{len(rows)} rows")
    check({Path(r["src"]).name for r in rows} == {"broken.bmp", "empty.bmp"},
          "[failures] the rows name the files that failed",
          str([r["src"] for r in rows]))

    with log.open(newline="", encoding="utf-8") as fh:
        run_row = list(csv.DictReader(fh))[-1]
    ids = {r["run_id"] for r in rows}
    check(bool(run_row["run_id"]) and ids == {run_row["run_id"]},
          "[failures] run_id joins the run log to the failure log",
          f"run={run_row['run_id']!r} failures={sorted(ids)}")
    check(run_row["error_kinds"] == "OSError:1 unreadable:1"
          and run_row["failures_logged"] == "2",
          "[failures] the run row summarises what the failure log details",
          f"error_kinds={run_row['error_kinds']!r} "
          f"failures_logged={run_row['failures_logged']!r}")

    run(src, tmp / "broken_out2", "bmp", log_path=log)
    with fail_log.open(newline="", encoding="utf-8") as fh:
        again = list(csv.DictReader(fh))
    check(len(again) == 4 and again[-1]["run_id"] not in ids,
          "[failures] a second failing run appends under its own run_id",
          f"{len(again)} rows, ids={sorted({r['run_id'] for r in again})}")

    # The extra file is a cost, so a clean history must not pay it.
    clean_src = tmp / "clean_input"
    clean_src.mkdir()
    Image.fromarray(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8),
                    "RGB").save(clean_src / "fine.bmp")
    clean_log = tmp / "clean_case" / "runs.csv"
    clean = run(clean_src, tmp / "clean_out", "bmp", log_path=clean_log)
    check(clean.failed == 0 and not gc.failure_log_path(clean_log).exists(),
          "[failures] a run with no failures writes no failure log",
          f"failed={clean.failed}")


# ---------------------------------------------------------------------------


def main() -> int:
    # Same reason as in ``app.main``: the installed ``rgb2gray-selftest``
    # console script calls this directly, bypassing ``__main__``.
    mp.freeze_support()

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="do not delete the fixture")
    args = ap.parse_args()

    gc.configure_start_method()
    tmp = Path(tempfile.mkdtemp(prefix="bmp2gray_selftest_"))
    src = tmp / "input"
    build_fixture(src)

    outs = {}
    for fmt in gc.OUTPUT_FORMATS:
        out = tmp / f"out_{fmt}"
        outs[fmt] = out
        # Pillow explicitly: these outputs are check_encoders' baseline, and
        # default_encoder() now picks OpenCV for PNG.
        st = run(src, out, fmt, encoder="pillow", log_path=out / "runs.csv")
        check_structure(src, out, st, fmt)
        check_annotation(src, out, fmt)
    check_shadowing(src, tmp)
    check_encoders(src, tmp, outs)
    check_shared_default(src, tmp)
    check_format_defaults(src, tmp)
    check_levels(src, tmp)
    check_lossless(outs)
    check_gpu(src, tmp, outs["bmp"])
    check_cancel(src, tmp)
    check_log(src, tmp, outs["bmp"] / "runs.csv")
    check_failure_log(tmp)

    failed = [r for r in RESULTS if not r[0]]
    width = max(len(name) for _ok, name, _d in RESULTS)
    for ok, name, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        line = f"  {mark}  {name:<{width}}"
        if detail and (not ok or detail == "skipped"):
            line += f"   {detail}"
        print(line)
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")

    if args.keep:
        print(f"fixture kept at {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
