"""
gray_core.py -- backend engine for the image -> 8-bit grayscale batch converter.

Walks an input tree, finds every readable image **by content** rather than by
extension, mirrors the folder structure exactly (empty directories included),
converts each image to 8-bit grayscale in the chosen format, and carries each
paired ``<same-stem>.json`` annotation across -- patching its ``imagePath`` only
when the output extension differs.

Two modes:

  CPU  ``multiprocessing.Pool``, one whole file per task.  The fast path.
  GPU  reader threads -> a batching device thread -> writer threads.  Kept for
       comparison; on the reference hardware it never won.  See REPORT.md.

Both use the ITU-R BT.601 luma weights ``Y = 0.299R + 0.587G + 0.114B``.  The
two paths are not bit-identical -- Pillow uses fixed-point integers, the device
path float32 -- but measured over full frames they never differ by more than
1 LSB.

The module is GUI-free on purpose: under ``spawn``/``forkserver`` every worker
process re-imports whatever module owns the worker function, and re-importing
``app.py`` would open a window per worker.  For the same reason ``torch`` is
imported lazily (:func:`torch_module`) -- CPU mode must not pay for it.

Requires Pillow and NumPy.  PyTorch only for GPU mode.
"""

from __future__ import annotations

import csv
import functools
import json
import multiprocessing as mp
import os
import platform
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import PIL
from PIL import Image, UnidentifiedImageError


# ---------------------------------------------------------------------------
# PyTorch is imported lazily
# ---------------------------------------------------------------------------
#
# Importing torch costs ~1 s and hundreds of MB, and every ``forkserver`` child
# imports this module to unpickle the worker function -- so a module-level
# import charged that second to CPU-mode start-up, inside the timed region.
# Measured: pool spin-up went from 3704 ms to 409 ms without it.

_torch = None
_torch_import_failed = False


def torch_module():
    """Return the ``torch`` module, importing it on first use, or ``None``."""
    global _torch, _torch_import_failed
    if _torch is None and not _torch_import_failed:
        try:
            import torch  # noqa: PLC0415 -- deliberately deferred
            _torch = torch
        except Exception:
            _torch_import_failed = True
    return _torch


_cv2 = None
_cv2_import_failed = False


def cv2_module():
    """Return the ``cv2`` module, importing it on first use, or ``None``.

    Deferred for the same reason as torch: OpenCV is an optional *encoder*
    backend, and the default path never touches it.
    """
    global _cv2, _cv2_import_failed
    if _cv2 is None and not _cv2_import_failed:
        try:
            import cv2  # noqa: PLC0415 -- deliberately deferred
            _cv2 = cv2
        except Exception:
            _cv2_import_failed = True
    return _cv2


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------

ANNOTATION_SUFFIX = ".json"

#: Extensions worth opening.  Discovery uses this list because stat-ing a tree
#: is cheap and opening every file in it is not; the actual format is then
#: determined from the file's content when it is opened.
INPUT_SUFFIXES = frozenset({
    ".bmp", ".png", ".jpg", ".jpeg", ".jpe", ".tif", ".tiff",
    ".webp", ".gif", ".ppm", ".pgm", ".pnm", ".tga",
})


#: Who does the *encoding*.  Reading and format sniffing are always Pillow's:
#: OpenCV has no API that reports what a file actually was, and returns ``None``
#: instead of raising on a file it cannot read -- which would cost the app its
#: detect-by-content behaviour and its error messages.  Measured, keeping Pillow
#: on the read side costs 4 % (108 ms against 104 ms for an all-OpenCV path).
ENCODERS = ("pillow", "opencv")
DEFAULT_ENCODER = "pillow"

#: zlib strategies OpenCV exposes for PNG.  Pillow exposes none of them, which
#: is the main reason the OpenCV backend is worth having.  ``rle`` is the
#: default because across all 40 PNG settings (2 encoders x 3
#: strategies x 10 levels) on 164 frames it was at once the *smallest* and the
#: *fastest* -- 28 % smaller and 1.22x faster than Pillow's level 1, and 8 %
#: smaller than Pillow's level 9 for 1/35th of its time.  Under ``rle`` levels
#: **1-9** give byte-identical output.  Level 0 is the exception: it stores
#: rather than deflates, so no strategy applies and the file comes out 4.6x
#: larger (786 MB against 169 MB over the reference 164 frames).  Since the UI
#: greys the level out under ``rle`` and calls it ignored, resolve_format lifts
#: a 0 to 1 there rather than let a disabled control quietly cost 4.6x.
PNG_STRATEGIES = ("default", "rle", "filtered")
DEFAULT_PNG_STRATEGY = "rle"


def detect_encoder() -> tuple[bool, str]:
    """``(opencv_usable, reason_or_version)`` -- safe to show in the UI."""
    cv2 = cv2_module()
    if cv2 is None:
        return False, "OpenCV is not installed - Pillow is the only encoder"
    return True, f"OpenCV {cv2.__version__}"


@dataclass(frozen=True)
class LevelKnob:
    """The one numeric compression control a format exposes, if it has any.

    Only PNG and JPEG do.  BMP is uncompressed, and Pillow's TIFF writer accepts
    ``compress_level`` and then ignores it, so offering a knob there would be a
    lie.  See README for the measurements.
    """

    key: str        # the Pillow ``save()`` keyword this control writes
    label: str      # what the number means, shown in the UI
    lo: int
    hi: int
    default: int
    hint: str       # what the two ends of the range do

    def clamp(self, value: int | None) -> int:
        return self.default if value is None else max(self.lo, min(self.hi, int(value)))


@dataclass(frozen=True)
class OutputFormat:
    """One selectable output format.

    ``lossless`` is surfaced in the UI rather than left for the reader to know:
    JPEG's artefacts land on the same scale as the smallest defects in this
    dataset.
    """

    key: str
    label: str
    pillow_format: str
    suffix: str
    save_kwargs: dict          # keywords that do not depend on the level
    lossless: bool
    note: str
    level: LevelKnob | None = None
    # Filled in per run by resolve_format; the table itself always holds the
    # defaults, because OUTPUT_FORMATS is shared across worker processes.
    encoder: str = DEFAULT_ENCODER
    png_strategy: str = DEFAULT_PNG_STRATEGY

    @property
    def effective_level(self) -> int | None:
        """The level this format will hand the encoder, once resolved."""
        return self.save_kwargs.get(self.level.key) if self.level else None


OUTPUT_FORMATS: dict[str, OutputFormat] = {
    # Defaults picked from one 2048x2448 frame, 5.01 MB as raw 8-bit grey:
    # PNG level 1 gives 1.35 MB in 118 ms, level 9 gives 1.03 MB in 4272 ms --
    # 24 % smaller for 36x the time, so 1.  Full table in README.
    "bmp": OutputFormat(
        "bmp", "BMP", "BMP", ".bmp", {}, True,
        "lossless · fastest to write, largest files",
    ),
    "png": OutputFormat(
        "png", "PNG", "PNG", ".png", {}, True,
        "lossless · ~4.6x smaller, ~3.7x slower",
        LevelKnob("compress_level", "PNG compression", 0, 9, 1,
                  "0 = store, 9 = smallest, all lossless"),
    ),
    "tiff": OutputFormat(
        "tiff", "TIFF", "TIFF", ".tiff", {"compression": "tiff_lzw"}, True,
        "lossless · ~3.8x smaller, ~2.6x slower",
    ),
    "jpeg": OutputFormat(
        "jpeg", "JPEG", "JPEG", ".jpg", {}, False,
        "LOSSY · artefacts are the size of small defects",
        LevelKnob("quality", "JPEG quality", 1, 100, 95,
                  "higher = larger, never lossless"),
    ),
}
DEFAULT_OUTPUT_FORMAT = "bmp"



def resolve_format(out_format: str, level: int | None = None, *,
                   encoder: str | None = None,
                   png_strategy: str = DEFAULT_PNG_STRATEGY) -> OutputFormat:
    """Everything one run needs in order to write a file, in one object.

    Returns a copy: :data:`OUTPUT_FORMATS` is shared, and in CPU mode it is
    re-imported in every worker process, so it must never be mutated per run.
    The copy is picklable, which is what lets the pool worker take it directly
    instead of re-resolving the settings itself.

    ``level`` is ignored for formats with no control, and clamped otherwise.
    ``encoder`` of ``None`` means :data:`DEFAULT_ENCODER`.  Under PNG's ``rle``
    strategy a level of 0 is lifted to 1 -- see :data:`PNG_STRATEGIES`.  A
    setting that cannot reach an encoder is blanked, so that what this returns
    is exactly what the run log, the GUI and the file all report.
    """
    fmt = OUTPUT_FORMATS[out_format]
    encoder = encoder if encoder in ENCODERS else DEFAULT_ENCODER
    strategy = (png_strategy if png_strategy in PNG_STRATEGIES
                else DEFAULT_PNG_STRATEGY)
    # A strategy exists only in OpenCV's PNG encoder.  Blank it everywhere else,
    # so the run log can never report a setting that reached no encoder: a BMP
    # run used to record whatever the PNG control happened to be left on.
    if not (out_format == "png" and encoder == "opencv"):
        strategy = ""

    kwargs = fmt.save_kwargs
    if fmt.level is not None:
        value = fmt.level.clamp(level)
        # rle caps zlib's match distance at 1, so levels 1-9 emit identical
        # bytes -- but level 0 still *stores*, 4.6x larger.  No caller can mean
        # "compress with rle, but do not compress", and the UI greys the control
        # out under rle, so a stale 0 would apply silently.  Floor it.
        if strategy == "rle":
            value = max(1, value)
        kwargs = {**kwargs, fmt.level.key: value}
    return replace(fmt, save_kwargs=kwargs, encoder=encoder,
                   png_strategy=strategy)


#: BT.601 luma weights in **RGB** order, matching :func:`_decode_rgb`.
LUMA_RGB = (0.299, 0.587, 0.114)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvertTask:
    """One source image plus its optional paired annotation.

    Pickled and shipped to worker processes, so every field must stay picklable.
    """

    src: Path                     # absolute path of the source image
    dst: Path                     # absolute path of the destination image
    ann_src: Path | None = None   # paired .json next to ``src``, if any
    ann_dst: Path | None = None   # where that .json has to land


@dataclass
class FileResult:
    """Outcome of a single file, produced by a worker and folded into RunStats."""

    src: str
    ok: bool
    error: str | None = None
    # A short, stable label for what went wrong -- "decode:OSError", "write:
    # OSError", "unreadable".  Set where the failure is raised, never parsed
    # back out of ``error``: the messages are free text and one of them carries
    # no exception name at all.
    error_kind: str = ""
    bytes_in: int = 0
    bytes_out: int = 0
    ann_copied: bool = False
    ann_rewritten: bool = False   # imagePath patched because the suffix changed
    src_format: str = ""          # what the file actually turned out to be


@dataclass
class RunStats:
    """Live, aggregated counters for one conversion run."""

    total: int = 0
    done: int = 0
    ok: int = 0
    failed: int = 0
    ann_copied: int = 0
    ann_rewritten: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    started: float = 0.0
    finished: float | None = None
    estimated_s: float | None = None   # prediction made early, kept for the log
    _ramp: float | None = None         # elapsed at ESTIMATE_FROM, see absorb()
    cancelled: bool = False
    mode: str = ""
    device: str = ""
    workers: int = 0                   # pipeline width -- comparable across modes
    threads: int = 0                   # what that width cost: processes or threads
    batch_size: int = 0
    out_format: str = ""
    level: int | None = None           # effective compression level, if any
    encoder: str = ""                  # pillow | opencv
    png_strategy: str = ""             # only meaningful for PNG via OpenCV
    formats_seen: dict[str, int] = field(default_factory=dict)
    #: ``(src, kind, message)`` per failed file, capped -- the detail the
    #: failure log writes.  Bounded because a run can fail on every one of a
    #: million files; :attr:`error_kinds` is what stays complete.
    errors: list[tuple[str, str, str]] = field(default_factory=list)
    #: Every failure counted by kind, never truncated: cardinality is the
    #: number of distinct faults, not the number of files.
    error_kinds: dict[str, int] = field(default_factory=dict)
    log_error: str | None = None       # why the run log was not written, if so
    failures_path: str | None = None   # the failure log, once one exists
    failures_error: str | None = None  # why it does not, if it should
    #: Identifies this run in both logs -- the run log has one row per run, the
    #: failure log many, and this is what joins them.  Second-resolution time
    #: plus 4 random hex: sortable, and two runs in the same second still differ.
    run_id: str = field(default_factory=lambda:
                        time.strftime("%Y%m%dT%H%M%S") + "-" + os.urandom(2).hex())

    #: The two sample points the duration prediction is taken from.  Two rather
    #: than one because a single point divides the *fixed* start-up cost into
    #: the rate and then multiplies it by every remaining image: measured over
    #: 438 runs, that over-predicted by a median 39 %, and no run shorter than
    #: 5 s landed within 10 %.  Differencing the two points cancels the constant
    #: and leaves the marginal per-image cost.
    ESTIMATE_FROM = 25
    ESTIMATE_AFTER = 100

    #: How many failures keep their per-file detail.  The failure log is meant
    #: to be acted on -- re-run these, drop those -- and a list longer than this
    #: is a broken run, not a work list.  ``failed`` and ``error_kinds`` stay
    #: exact past the cap, and ``failures_logged`` says where it bit.
    MAX_LOGGED_FAILURES = 10_000

    # -- derived ----------------------------------------------------------
    @property
    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.perf_counter()) - self.started

    @property
    def rate(self) -> float:
        """Images per second, over the whole run including start-up."""
        e = self.elapsed
        return self.ok / e if e > 0 and self.ok else 0.0

    @property
    def read_mb_s(self) -> float:
        e = self.elapsed
        return (self.bytes_in / 1e6) / e if e > 0 else 0.0

    @property
    def write_mb_s(self) -> float:
        e = self.elapsed
        return (self.bytes_out / 1e6) / e if e > 0 else 0.0

    def absorb(self, r: FileResult) -> None:
        """Fold a single :class:`FileResult` into the running totals."""
        self.done += 1
        if r.ok:
            self.ok += 1
            self.bytes_in += r.bytes_in
            self.bytes_out += r.bytes_out
            if r.src_format:
                self.formats_seen[r.src_format] = \
                    self.formats_seen.get(r.src_format, 0) + 1
        else:
            self.failed += 1
            kind = r.error_kind or "unknown"
            self.error_kinds[kind] = self.error_kinds.get(kind, 0) + 1
            if len(self.errors) < self.MAX_LOGGED_FAILURES:
                self.errors.append((r.src, kind, r.error or "unknown error"))
        if r.ann_copied:
            self.ann_copied += 1
        if r.ann_rewritten:
            self.ann_rewritten += 1

        # Predict the total once, early, so the log can compare the prediction
        # against the outcome instead of quoting a hindsight figure.  Runs
        # shorter than ESTIMATE_AFTER get no prediction at all, which is the
        # honest answer: they finish before the pipeline is even warm.
        if self.done == self.ESTIMATE_FROM:
            self._ramp = self.elapsed
        elif self.estimated_s is None and self.done == self.ESTIMATE_AFTER:
            e = self.elapsed
            span = self.ESTIMATE_AFTER - self.ESTIMATE_FROM
            if e > 0 and self._ramp is not None:
                per = max(0.0, (e - self._ramp) / span)
                self.estimated_s = e + per * (self.total - self.done)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "device": self.device,
            "workers": self.workers,
            "os_workers": self.threads,
            "batch_size": self.batch_size,
            "output_format": self.out_format,
            "compression_level": self.level,
            "encoder": self.encoder,
            "png_strategy": self.png_strategy,
            "total": self.total,
            "converted": self.ok,
            "failed": self.failed,
            "annotations_copied": self.ann_copied,
            "annotations_rewritten": self.ann_rewritten,
            "source_formats": dict(self.formats_seen),
            "estimated_s": round(self.estimated_s, 2) if self.estimated_s else None,
            "elapsed_s": round(self.elapsed, 3),
            "images_per_s": round(self.rate, 2),
            "read_MB_s": round(self.read_mb_s, 1),
            "write_MB_s": round(self.write_mb_s, 1),
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "cancelled": self.cancelled,
        }


#: ``progress_cb(result, stats)`` -- called once per finished file.  In CPU mode
#: it runs on the caller's thread; in GPU mode on a writer thread, so the GUI
#: must marshal it back onto the Tk thread itself (see ``app.py``).
ProgressCB = Callable[[FileResult, RunStats], None]


# ---------------------------------------------------------------------------
# Machine description (for the run log)
# ---------------------------------------------------------------------------


def physical_cores() -> int:
    """Best-effort count of *physical* cores, not hyper-threads."""
    try:  # Linux: count distinct (physical id, core id) pairs
        pairs, phys, core = set(), None, None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
                if phys is not None:
                    pairs.add((phys, core))
        if pairs:
            return len(pairs)
    except Exception:
        pass
    try:  # Linux fallback: is SMT on at all?
        if Path("/sys/devices/system/cpu/smt/active").read_text().strip() == "1":
            return max(1, (os.cpu_count() or 2) // 2)
    except Exception:
        pass
    return os.cpu_count() or 1


def default_workers() -> int:
    """Default **pipeline width**: how many images are worked on in parallel.

    It means the same thing in both modes -- which is what makes a CPU run and a
    GPU run comparable.  Only the cost differs: CPU mode has one stage per
    image, so width N is N processes; GPU mode has two CPU-side stages (decode
    and encode, the conversion having moved to the device), so width N is 2N
    threads.
    """
    return physical_cores()


def nvidia_smi() -> list[dict]:
    """Identity of each NVIDIA GPU, or ``[]`` when ``nvidia-smi`` is unavailable.

    Only identity: live utilisation is sampled before the run starts and would
    always read as idle, so it is not worth a column.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "driver": parts[1],
                         "memory_total_MiB": parts[2]})
    return gpus


def machine_info() -> dict:
    """Everything about the host worth recording next to a timing."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": physical_cores(),
    }
    try:  # /proc is Linux-only; fall back silently elsewhere
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except Exception:
        pass
    torch = torch_module()
    info["pillow"] = PIL.__version__
    info["numpy"] = np.__version__
    info["torch"] = torch.__version__ if torch is not None else None
    info["gpus"] = nvidia_smi()
    return info


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------


def detect_gpu() -> tuple[str | None, str]:
    """Return ``(device_string, human_readable_reason_or_name)``.

    ``device_string`` is ``"cuda"``, ``"mps"`` or ``None`` when no accelerator is
    usable.  The second element is always safe to show in the UI.
    """
    torch = torch_module()
    if torch is None:
        return None, "PyTorch is not installed - GPU mode unavailable"
    try:
        if torch.cuda.is_available():
            return "cuda", f"CUDA: {torch.cuda.get_device_name(0)}"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "Apple Metal (MPS)"
    except Exception as exc:  # a broken driver must not crash the GUI
        return None, f"GPU probe failed: {exc}"
    return None, f"torch {torch.__version__} sees no CUDA/MPS device (CPU-only build?)"


def resolve_device(spec: str = "auto"):
    """Turn a device spec into a ``torch.device``.

    ``"auto"`` picks the accelerator and fails loudly when there is none.  An
    explicit ``"cpu"`` is honoured, which runs the batched tensor pipeline with
    no accelerator behind it -- the only way to exercise that code path on a
    machine that has no GPU.
    """
    torch = torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed; GPU mode is unavailable.")
    if spec == "auto":
        dev, reason = detect_gpu()
        if dev is None:
            raise RuntimeError(reason)
        return torch.device(dev)
    return torch.device(spec)


# ---------------------------------------------------------------------------
# Filesystem discovery / mirroring
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    tasks: list[ConvertTask]
    directories: list[Path]           # every sub-directory, relative to the input root
    orphan_annotations: list[Path]    # .json with no same-stem image (reported only)
    skipped_files: list[Path]         # files whose extension is not an image one
    already_done: int                 # images skipped because the output existed
    suffixes_seen: dict[str, int]     # input extension -> count (queued images only)
    shadowed: list[tuple[Path, str]]  # (source, the output name another source claimed)


def scan_input(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    out_format: str = DEFAULT_OUTPUT_FORMAT,
    skip_existing: bool = False,
) -> ScanResult:
    """Walk ``input_dir`` and build the full work list.

    * Recursive, case-insensitive match against :data:`INPUT_SUFFIXES`.
    * Every sub-directory is recorded so the output tree can mirror the input
      tree *exactly*, including folders that happen to hold no images.
    * A ``.json`` sitting next to ``foo.bmp`` and named ``foo.json`` is paired
      with it and travels to the same relative location.
    * If the output directory lives inside the input directory it is pruned from
      the walk, otherwise a re-run would try to convert its own output.
    """
    fmt = OUTPUT_FORMATS[out_format]
    in_root = Path(input_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    if not in_root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {in_root}")

    tasks: list[ConvertTask] = []
    directories: list[Path] = []
    orphans: list[Path] = []
    skipped: list[Path] = []
    shadowed: list[tuple[Path, str]] = []
    suffixes: dict[str, int] = {}
    already_done = 0

    for dirpath, dirnames, filenames in os.walk(in_root):
        here = Path(dirpath)

        # Never descend into the output tree (handles out_root nested in in_root).
        dirnames[:] = [d for d in dirnames if (here / d).resolve() != out_root]
        dirnames.sort()

        rel_dir = here.relative_to(in_root)
        if rel_dir != Path("."):
            directories.append(rel_dir)

        stems_with_image: set[str] = set()
        claimed: set[str] = set()   # output names already spoken for in this folder
        for name in sorted(filenames):
            src = here / name
            suffix = src.suffix.lower()
            if suffix == ANNOTATION_SUFFIX:
                continue
            if suffix not in INPUT_SUFFIXES:
                skipped.append(src.relative_to(in_root))
                continue

            stems_with_image.add(src.stem)

            # Many input formats, one output format: photo.png and photo.jpg
            # both want photo.bmp.  Resolve it here, once -- first in sorted
            # order takes the name, the rest are dropped and reported.  Letting
            # the workers race for it gave different winners run to run.
            out_name = src.stem + fmt.suffix
            if out_name in claimed:
                shadowed.append((src.relative_to(in_root), out_name))
                continue
            claimed.add(out_name)

            suffixes[suffix] = suffixes.get(suffix, 0) + 1
            dst = out_root / rel_dir / out_name

            if skip_existing and dst.exists():
                already_done += 1
                continue

            ann_src = src.with_suffix(ANNOTATION_SUFFIX)
            if ann_src.exists():
                ann_dst = dst.with_suffix(ANNOTATION_SUFFIX)
            else:
                ann_src, ann_dst = None, None
            tasks.append(ConvertTask(src=src, dst=dst, ann_src=ann_src, ann_dst=ann_dst))

        # Any .json whose same-stem image is missing would otherwise be dropped.
        for name in filenames:
            if name.lower().endswith(ANNOTATION_SUFFIX):
                p = here / name
                if p.stem not in stems_with_image:
                    orphans.append(p.relative_to(in_root))

    return ScanResult(
        tasks=tasks,
        directories=directories,
        orphan_annotations=orphans,
        skipped_files=skipped,
        already_done=already_done,
        suffixes_seen=suffixes,
        shadowed=shadowed,
    )


def sweep_temp_files(output_dir: str | Path) -> int:
    """Delete staging files an interrupted run left behind.

    Cancelling SIGTERMs the pool, which can kill a worker between writing the
    temp file and renaming it into place.
    """
    removed = 0
    for path in Path(output_dir).rglob(".*.part*"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def mirror_tree(output_dir: str | Path, directories: Iterable[Path]) -> None:
    """Pre-create the whole directory skeleton in one pass.

    Keeps ``mkdir`` out of the hot loop, and mirrors folders holding no images.
    """
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for rel in directories:
        (out_root / rel).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Low-level image and annotation helpers
# ---------------------------------------------------------------------------


def _tmp_path(dst: Path) -> Path:
    """Sibling scratch name used for atomic writes (same filesystem => rename is atomic)."""
    return dst.with_name(f".{dst.name}.part{os.getpid()}")


def _decode_rgb(src: Path) -> tuple[np.ndarray, str]:
    """Decode any supported image into ``(H, W, 3)`` uint8 **RGB**, plus its format.

    Only GPU mode needs this: the device wants real arrays to stack into a batch.
    CPU mode never builds an array at all.
    """
    with Image.open(src) as im:
        fmt = im.format or "?"
        return np.asarray(im.convert("RGB")), fmt


def _cv_encode(gray: np.ndarray, fmt: OutputFormat) -> bytes:
    """OpenCV's encoder, as bytes.

    ``imencode`` rather than ``imwrite`` because the atomic write below targets
    a hidden ``.part`` file and OpenCV picks its encoder from the extension.
    Measured identical in both bytes and time to writing the file directly.
    """
    cv2 = cv2_module()
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed")
    params: list[int] = []
    if fmt.key == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, fmt.effective_level or 0]
        strategy = {
            "default": cv2.IMWRITE_PNG_STRATEGY_DEFAULT,
            "rle": cv2.IMWRITE_PNG_STRATEGY_RLE,
            "filtered": cv2.IMWRITE_PNG_STRATEGY_FILTERED,
        }.get(fmt.png_strategy)
        if strategy is not None:
            params += [cv2.IMWRITE_PNG_STRATEGY, strategy]
    elif fmt.key == "jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, fmt.effective_level or 95]
    elif fmt.key == "tiff":                       # match Pillow's tiff_lzw
        # 5 is libtiff's COMPRESSION_LZW.  The named constant only arrived in
        # OpenCV 4.10, while the parameter itself has worked since long before,
        # so the literal is what keeps an older wheel writing LZW instead of
        # failing every TIFF in the run.
        lzw = getattr(cv2, "IMWRITE_TIFF_COMPRESSION_LZW", 5)
        params = [cv2.IMWRITE_TIFF_COMPRESSION, lzw]
    ok, buf = cv2.imencode(fmt.suffix, gray, params)
    if not ok:
        raise RuntimeError(f"OpenCV could not encode {fmt.suffix}")
    return buf.tobytes()


def _save_atomic(dst: Path, fmt: OutputFormat, *,
                 pil: Image.Image | None = None,
                 gray: np.ndarray | None = None) -> None:
    """Encode into ``dst`` through a hidden sibling plus ``os.replace``.

    Cancelling SIGTERMs the pool workers, so an in-place write could leave a
    truncated file that later looks like a valid conversion.

    The caller passes whichever shape it already has: CPU mode has a PIL image
    (and never builds an array -- that is its fast path), GPU mode has the
    array.  Only the backend that needs the other shape pays for the
    conversion, which is why both arguments exist.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(dst)
    try:
        if fmt.encoder == "opencv":
            if gray is None:
                gray = np.asarray(pil)
            tmp.write_bytes(_cv_encode(gray, fmt))
        else:
            if pil is None:
                pil = Image.fromarray(gray, mode="L")
            pil.save(tmp, format=fmt.pillow_format, **fmt.save_kwargs)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


#: Matches the ``imagePath`` member of a LabelMe/X-AnyLabeling annotation without
#: disturbing anything else in the document.
_IMAGE_PATH_RE = re.compile(r'("imagePath"\s*:\s*)("(?:[^"\\]|\\.)*")')


def _sync_annotation(task: ConvertTask) -> tuple[bool, bool]:
    """Copy the paired ``.json``.  Returns ``(copied, rewritten)``.

    Same extension in and out: a verbatim ``shutil.copy2``.  Different extension:
    the annotation's ``imagePath`` would name a file that is not there, so that
    one member is rewritten by regex on the raw text -- indentation, key order
    and every other byte survive, which a parse/re-serialise round trip would
    not guarantee.  Anything unexpected falls back to the verbatim copy.
    """
    if task.ann_src is None or task.ann_dst is None:
        return False, False

    task.ann_dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(task.ann_dst)
    rewritten = False
    try:
        if task.src.suffix.lower() != task.dst.suffix.lower():
            try:
                text = task.ann_src.read_text(encoding="utf-8")
                matches = _IMAGE_PATH_RE.findall(text)
                if len(matches) == 1:
                    old = json.loads(matches[0][1])
                    new = str(Path(old).with_suffix(task.dst.suffix))
                    if new != old:
                        text = _IMAGE_PATH_RE.sub(
                            lambda m: m.group(1) + json.dumps(new, ensure_ascii=False),
                            text, count=1)
                        tmp.write_text(text, encoding="utf-8")
                        shutil.copystat(task.ann_src, tmp)
                        rewritten = True
            except (OSError, UnicodeDecodeError, ValueError):
                rewritten = False  # fall through to the verbatim copy

        if not rewritten:
            shutil.copy2(task.ann_src, tmp)
        os.replace(tmp, task.ann_dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return True, rewritten


# ---------------------------------------------------------------------------
# CPU mode
# ---------------------------------------------------------------------------


def cpu_convert_one(task: ConvertTask, fmt: OutputFormat) -> FileResult:
    """Convert exactly one image.  Runs inside a worker process in CPU mode.

    Must stay at module level with picklable arguments so ``spawn``/
    ``forkserver`` children can import and call it; ``fmt`` comes from
    :func:`resolve_format` and is picklable for exactly that reason.

    ``convert("L")`` fuses decode and luma in one C loop and never builds a
    numpy array -- the fast path, which only the OpenCV encoder gives up.
    """
    src_str = str(task.src)
    try:
        bytes_in = task.src.stat().st_size
        with Image.open(task.src) as im:
            src_format = im.format or "?"
            _save_atomic(task.dst, fmt, pil=im.convert("L"))
        # Mirror timestamps/permissions, exactly as copy2 does for the .json.
        shutil.copystat(task.src, task.dst)

        bytes_out = task.dst.stat().st_size
        copied, rewritten = _sync_annotation(task)
        return FileResult(
            src=src_str, ok=True, bytes_in=bytes_in, bytes_out=bytes_out,
            ann_copied=copied, ann_rewritten=rewritten, src_format=src_format,
        )
    except UnidentifiedImageError:
        return FileResult(src=src_str, ok=False, error_kind="unreadable",
                          error="not a readable image (content does not match any "
                                "known format)")
    except Exception as exc:
        return FileResult(src=src_str, ok=False,
                          error_kind=type(exc).__name__,
                          error=f"{type(exc).__name__}: {exc}")


def run_cpu(
    tasks: Sequence[ConvertTask],
    *,
    out_format: str = DEFAULT_OUTPUT_FORMAT,
    level: int | None = None,
    encoder: str | None = None,
    png_strategy: str = DEFAULT_PNG_STRATEGY,
    workers: int | None = None,
    progress_cb: ProgressCB | None = None,
    cancel: threading.Event | None = None,
    stats: RunStats | None = None,
) -> RunStats:
    """Fan the task list out over a process pool, one whole file per task."""
    fmt = resolve_format(out_format, level, encoder=encoder,
                         png_strategy=png_strategy)
    workers = int(workers or default_workers())
    workers = max(1, min(workers, len(tasks) or 1))

    st = stats or RunStats()
    st.mode = "cpu"
    st.device = f"{workers} process(es)"
    st.workers = st.threads = workers      # one stage, so width == process count
    st.out_format = out_format
    st.level = fmt.effective_level
    st.encoder, st.png_strategy = fmt.encoder, fmt.png_strategy
    st.total = len(tasks)
    st.started = time.perf_counter()

    if not tasks:
        st.finished = time.perf_counter()
        return st

    worker = functools.partial(cpu_convert_one, fmt=fmt)

    # ``chunksize=1`` keeps the progress bar smooth.  The per-task IPC overhead
    # is irrelevant next to milliseconds of work per image.
    with mp.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(worker, tasks, chunksize=1):
            st.absorb(result)
            if progress_cb is not None:
                progress_cb(result, st)
            if cancel is not None and cancel.is_set():
                st.cancelled = True
                pool.terminate()
                break

    st.finished = time.perf_counter()
    return st


# ---------------------------------------------------------------------------
# GPU mode -- reader threads -> batching device thread -> writer threads
# ---------------------------------------------------------------------------


def _q_put(q: "queue.Queue", item, cancel: threading.Event) -> bool:
    """Blocking put that still honours cancellation.  False == cancelled."""
    while True:
        if cancel.is_set():
            return False
        try:
            q.put(item, timeout=0.2)
            return True
        except queue.Full:
            continue


def _q_get(q: "queue.Queue", cancel: threading.Event):
    """Blocking get that still honours cancellation.  Returns ``_CANCELLED``."""
    while True:
        if cancel.is_set():
            return _CANCELLED
        try:
            return q.get(timeout=0.2)
        except queue.Empty:
            continue


_CANCELLED = object()   # sentinel: the run was aborted
_SENTINEL = object()    # sentinel: no more items on this queue


def gpu_gray_batch(batch: np.ndarray, device, weights) -> np.ndarray:
    """Convert a ``(B, H, W, 3)`` uint8 RGB batch to ``(B, H, W)`` uint8 luma."""
    torch = torch_module()
    t = torch.from_numpy(batch)
    if device.type == "cuda":
        # Pinned (page-locked) staging memory is what makes ``non_blocking``
        # transfers actually asynchronous, letting the copy overlap compute.
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        t = t.to(device)
    w = torch.tensor(weights, dtype=torch.float32, device=t.device)
    y = (t.to(torch.float32) @ w).round_().clamp_(0, 255).to(torch.uint8)
    return y.to("cpu").numpy()


def _encode_one(task: ConvertTask, gray: np.ndarray, nbytes: int,
                src_format: str, fmt: OutputFormat) -> FileResult:
    """Stage 3 for one frame: write the grey image and sync its ``.json``.

    Shared by the threaded writers and the serial path so the two cannot drift.
    """
    try:
        _save_atomic(task.dst, fmt, gray=gray)
        shutil.copystat(task.src, task.dst)
        bytes_out = task.dst.stat().st_size
        copied, rewritten = _sync_annotation(task)
        return FileResult(src=str(task.src), ok=True, bytes_in=nbytes,
                          bytes_out=bytes_out, ann_copied=copied,
                          ann_rewritten=rewritten, src_format=src_format)
    except Exception as exc:
        return FileResult(src=str(task.src), ok=False,
                          error_kind=f"write:{type(exc).__name__}",
                          error=f"write: {type(exc).__name__}: {exc}")


def _device_pass(items: list, dev, report) -> list:
    """Run one shape-uniform group of decoded frames through the device.

    ``items`` are ``(task, rgb_array, nbytes, src_format)`` and are **consumed**:
    the arrays are released as soon as the batch has been stacked, so peak memory
    stays at one batch.  Returns the same tuples with each array replaced by its
    8-bit luma result, or ``[]`` if the group was empty or the device call failed
    -- in the latter case every frame in it has already been reported as an error.
    """
    if not items:
        return []
    batch = np.stack([a for (_t, a, _n, _f) in items])
    meta = [(t, n, f) for (t, _a, n, f) in items]
    items.clear()
    try:
        grays = gpu_gray_batch(batch, dev, LUMA_RGB)
    except Exception as exc:
        msg = f"{dev.type}: {type(exc).__name__}: {exc}"
        kind = f"{dev.type}:{type(exc).__name__}"
        for task, _n, _f in meta:
            report(FileResult(src=str(task.src), ok=False,
                              error_kind=kind, error=msg))
        return []
    return [(t, grays[i], n, f) for i, (t, n, f) in enumerate(meta)]


def _gpu_serial(tasks, fmt, dev, batch_size, report, cancel, st) -> None:
    """GPU mode with no threads at all: decode, device, encode, in order.

    The baseline the threaded pipeline is measured against, selected by
    ``workers=1``.  Batching still applies, but only across frames that arrive
    consecutively at the same resolution -- with nothing running in parallel
    there is no device to keep fed, so holding frames back would only inflate
    peak memory.
    """
    pending: list = []

    def flush() -> None:
        for done in _device_pass(pending, dev, report):
            report(_encode_one(*done, fmt))

    for task in tasks:
        if cancel.is_set():
            st.cancelled = True
            return
        try:
            nbytes = task.src.stat().st_size
            arr, src_format = _decode_rgb(task.src)
        except Exception as exc:
            report(FileResult(src=str(task.src), ok=False,
                              error_kind=f"decode:{type(exc).__name__}",
                              error=f"decode: {type(exc).__name__}: {exc}"))
            continue
        if pending and arr.shape[:2] != pending[0][1].shape[:2]:
            flush()                      # resolution changed: close the batch
        pending.append((task, arr, nbytes, src_format))
        if len(pending) >= batch_size:
            flush()
    if cancel.is_set():
        st.cancelled = True
    else:
        flush()


def run_gpu(
    tasks: Sequence[ConvertTask],
    *,
    out_format: str = DEFAULT_OUTPUT_FORMAT,
    level: int | None = None,
    encoder: str | None = None,
    png_strategy: str = DEFAULT_PNG_STRATEGY,
    device: str = "auto",
    batch_size: int = 1,
    workers: int | None = None,
    progress_cb: ProgressCB | None = None,
    cancel: threading.Event | None = None,
    stats: RunStats | None = None,
) -> RunStats:
    """Batched tensor pipeline.  Kept as a comparison mode -- see REPORT.md.

    Three stages joined by bounded queues, so memory stays flat however long the
    job is: ``readers`` threads decode to uint8 RGB, this thread buckets frames
    **by shape** (a tensor batch needs a uniform size) and pushes each full
    bucket to the device, ``writers`` threads encode the results.

    ``workers`` is the pipeline width -- see :func:`default_workers` -- so it
    costs twice that many threads.  ``workers=1`` drops them entirely and runs
    :func:`_gpu_serial`.

    ``batch_size`` defaults to 1 because larger batches measured *monotonically
    worse* on the reference GPU (282.8 img/s at 1, 101.1 at 64): the staging
    copies land on the host memory path, which is the constraint.
    """
    fmt = resolve_format(out_format, level, encoder=encoder,
                         png_strategy=png_strategy)
    dev = resolve_device(device)
    cancel = cancel or threading.Event()

    # ``workers`` is the pipeline width, not a thread count: each unit of width
    # buys one decoder and one encoder, because those are the two CPU-side stages
    # left once the conversion itself moved to the device.  Two batches in flight
    # is enough to keep the device fed without letting frames pile up in RAM.
    width = (default_workers() if workers is None
             else max(1, int(workers)))
    width = max(1, min(width, len(tasks) or 1))   # as run_cpu does
    readers = writers = width
    batch_size = max(1, int(batch_size))
    prefetch = max(4, batch_size * 2)

    st = stats or RunStats()
    st.mode = "gpu"
    torch = torch_module()
    if dev.type == "cuda":
        st.device = f"cuda ({torch.cuda.get_device_name(0)})"
    else:
        st.device = dev.type
    st.workers = width
    st.threads = 1 if width == 1 else readers + writers
    st.batch_size = batch_size
    st.out_format = out_format
    st.level = fmt.effective_level
    st.encoder, st.png_strategy = fmt.encoder, fmt.png_strategy
    st.total = len(tasks)
    st.started = time.perf_counter()

    if not tasks:
        st.finished = time.perf_counter()
        return st

    lock = threading.Lock()

    def report(result: FileResult) -> None:
        """Thread-safe progress fan-in (called from reader and writer threads)."""
        with lock:
            st.absorb(result)
            if progress_cb is not None:
                progress_cb(result, st)

    if width == 1:
        _gpu_serial(tasks, fmt, dev, batch_size, report, cancel, st)
        st.finished = time.perf_counter()
        return st

    task_q: "queue.Queue" = queue.Queue()
    for t in tasks:
        task_q.put(t)
    dec_q: "queue.Queue" = queue.Queue(maxsize=prefetch)
    out_q: "queue.Queue" = queue.Queue(maxsize=prefetch)

    # -- stage 1: readers --------------------------------------------------
    def reader_loop() -> None:
        while not cancel.is_set():
            try:
                task = task_q.get_nowait()
            except queue.Empty:
                return
            try:
                nbytes = task.src.stat().st_size
                arr, src_format = _decode_rgb(task.src)
                if not _q_put(dec_q, (task, arr, nbytes, src_format), cancel):
                    return
            except Exception as exc:
                report(FileResult(src=str(task.src), ok=False,
                                  error_kind=f"decode:{type(exc).__name__}",
                                  error=f"decode: {type(exc).__name__}: {exc}"))

    reader_threads = [
        threading.Thread(target=reader_loop, name=f"reader-{i}", daemon=True)
        for i in range(readers)
    ]

    # -- stage 3: writers --------------------------------------------------
    def writer_loop() -> None:
        while True:
            item = _q_get(out_q, cancel)
            if item is _CANCELLED or item is _SENTINEL:
                return
            report(_encode_one(*item, fmt))

    writer_threads = [
        threading.Thread(target=writer_loop, name=f"writer-{i}", daemon=True)
        for i in range(writers)
    ]

    # A tiny helper thread closes ``dec_q`` once *every* reader has finished, so
    # the batcher below knows when to flush its partial buckets.
    def closer_loop() -> None:
        for th in reader_threads:
            th.join()
        _q_put(dec_q, _SENTINEL, cancel)

    closer = threading.Thread(target=closer_loop, name="closer", daemon=True)

    for th in reader_threads + writer_threads:
        th.start()
    closer.start()

    # -- stage 2: batching + device work (runs on this thread) -------------
    buckets: dict[tuple[int, int], list] = {}
    buffered = 0
    max_buffered = max(batch_size * 2, batch_size + 4)

    def flush(shape: tuple[int, int]) -> bool:
        """Convert one bucket and hand it to the writers.  False == cancelled."""
        nonlocal buffered
        items = buckets.pop(shape, None)
        if not items:
            return True
        buffered -= len(items)
        for done in _device_pass(items, dev, report):
            if not _q_put(out_q, done, cancel):
                return False
        return True

    try:
        while True:
            item = _q_get(dec_q, cancel)
            if item is _CANCELLED:
                st.cancelled = True
                break
            if item is _SENTINEL:
                break
            task, arr, nbytes, src_format = item
            shape = (arr.shape[0], arr.shape[1])
            buckets.setdefault(shape, []).append((task, arr, nbytes, src_format))
            buffered += 1

            if len(buckets[shape]) >= batch_size:
                if not flush(shape):
                    st.cancelled = True
                    break
            elif buffered >= max_buffered:
                # Mixed-resolution input: never let half-full buckets pile up.
                biggest = max(buckets, key=lambda s: len(buckets[s]))
                if not flush(biggest):
                    st.cancelled = True
                    break

        if not cancel.is_set():
            for shape in list(buckets):  # final partial batches
                if not flush(shape):
                    break
    finally:
        # Stop the stages in pipeline order: readers, then the batcher (already
        # unwound above), then the writers -- otherwise a writer could exit while
        # a reader is still feeding work into the queue behind it.
        for th in reader_threads:
            th.join(timeout=10.0)
        closer.join(timeout=5.0)
        for _ in writer_threads:
            try:
                out_q.put(_SENTINEL, timeout=1.0)
            except queue.Full:
                pass
        for th in writer_threads:
            th.join(timeout=30.0)
        if cancel.is_set():
            st.cancelled = True

    st.finished = time.perf_counter()
    return st


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------


def _histogram_cell(counts: dict) -> str:
    """Render a small mapping as one CSV cell: ``BMP:2 PNG:1``."""
    return " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


#: Column order of the run log.  Every key :func:`build_run_record` produces must
#: appear here -- ``DictWriter`` silently drops anything that does not, and
#: ``selftest.py`` asserts the two stay in step.
RUN_LOG_COLUMNS = (
    "run_id", "timestamp", "input_dir", "output_dir",
    # what was asked for
    "mode", "device", "workers", "os_workers", "batch_size", "output_format",
    "compression_level", "encoder", "png_strategy",
    # what came out
    "images", "converted", "failed", "cancelled",
    "annotations_copied", "annotations_rewritten", "source_formats",
    # timing
    "estimated_s", "elapsed_s", "estimate_time_error_pct", "images_per_s",
    "read_MB_s", "write_MB_s", "bytes_in", "bytes_out", "bytes_saved_pct",
    # what the scan found but did not convert
    "sub_folders", "unpaired_annotations", "non_image_files_skipped",
    "shadowed_by_same_name", "already_converted_skipped",
    # the machine the timing belongs to -- a timing without one means nothing
    "cpu", "physical_cores", "logical_cpus", "ram_gb", "platform",
    "python", "pillow", "numpy", "torch",
    "gpu", "gpu_driver", "gpu_memory_MiB",
    # what failed -- see the failure log, joined on run_id, for which files
    "error_kinds", "failures_logged", "first_error",
)


#: Column order of the failure log: one row per failed file, not per run.
FAILURE_LOG_COLUMNS = ("run_id", "timestamp", "src", "error_kind", "error")


def _append_csv(path: Path, columns: Sequence[str], rows: Iterable[dict]) -> None:
    """Append rows to a CSV, writing the header only for a new file.

    An existing file's own header wins over ``columns``, so appending to a log
    written by a build with a different column set drops the new columns instead
    of shifting every later row out of line.  That is also the upgrade path: an
    old log keeps its shape, and a new one is one new filename away.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header: Sequence[str] = columns
    fresh = True
    if path.exists() and path.stat().st_size:
        with path.open("r", encoding="utf-8", newline="") as fh:
            first = next(csv.reader(fh), None)
        if first:
            header, fresh = first, False
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header,
                                restval="", extrasaction="ignore")
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


def append_run_log(log_path: str | Path, record: dict) -> None:
    """Append one CSV row per run.

    CSV rather than JSON so the log opens in a spreadsheet; the price is that
    :func:`build_run_record` has to flatten everything to scalars first -- which
    is also why *which files* failed cannot live here, and gets a second file.
    """
    _append_csv(Path(log_path).expanduser(), RUN_LOG_COLUMNS, [record])


def failure_log_path(log_path: str | Path) -> Path:
    """Where the per-file failures of ``log_path``'s runs are recorded."""
    path = Path(log_path).expanduser()
    return path.with_name(f"{path.stem}_failures{path.suffix or '.csv'}")


def append_failure_log(log_path: str | Path, run_id: str, timestamp: str,
                       errors: Sequence[tuple[str, str, str]]) -> Path:
    """Append one row per failed file, and return the file written.

    A sibling of the run log rather than a column in it: the run log is one row
    per run and stays readable in a spreadsheet, while this is 0..n rows per
    run.  ``run_id`` is the join between the two.  Written only when a run
    actually failed, so a clean history never grows the extra file.
    """
    path = failure_log_path(log_path)
    _append_csv(path, FAILURE_LOG_COLUMNS,
                ({"run_id": run_id, "timestamp": timestamp, "src": src,
                  "error_kind": kind, "error": message}
                 for src, kind, message in errors))
    return path


def build_run_record(stats: RunStats, input_dir, output_dir,
                     scan: ScanResult) -> dict:
    """Assemble one flat log row for a finished run.

    Flat on purpose: the row is a CSV line, so every value here has to be a
    scalar.  Keys must match :data:`RUN_LOG_COLUMNS` exactly.
    """
    run = stats.as_dict()
    run["run_id"] = stats.run_id
    run["images"] = run.pop("total")               # "total" is ambiguous in a table
    run["source_formats"] = _histogram_cell(run["source_formats"])
    if run["compression_level"] is None:           # BMP/TIFF have no control
        run["compression_level"] = ""
    if not run["batch_size"]:                      # CPU mode has no batches
        run["batch_size"] = ""

    est = stats.estimated_s
    saved = stats.bytes_in - stats.bytes_out
    host = machine_info()
    gpus = host["gpus"]

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        **run,
        "estimate_time_error_pct": (round(100.0 * (stats.elapsed - est) / est, 1)
                                    if est else ""),
        "bytes_saved_pct": (round(100.0 * saved / stats.bytes_in, 2)
                            if stats.bytes_in else ""),
        "sub_folders": len(scan.directories),
        "unpaired_annotations": len(scan.orphan_annotations),
        "non_image_files_skipped": len(scan.skipped_files),
        "shadowed_by_same_name": len(scan.shadowed),
        "already_converted_skipped": scan.already_done,
        **{k: host.get(k) or "" for k in
           ("cpu", "physical_cores", "logical_cpus", "ram_gb",
            "platform", "python", "pillow", "numpy", "torch")},
        "gpu": "; ".join(g["name"] for g in gpus),
        "gpu_driver": gpus[0]["driver"] if gpus else "",
        "gpu_memory_MiB": gpus[0]["memory_total_MiB"] if gpus else "",
        "error_kinds": _histogram_cell(stats.error_kinds),
        # Not the same as `failed` once the cap bites: this is how many of them
        # reached the failure log, so a short list is never mistaken for a
        # complete one.
        "failures_logged": len(stats.errors),
        "first_error": (f"{stats.errors[0][0]}: {stats.errors[0][2]}"
                        if stats.errors else ""),
    }


# ---------------------------------------------------------------------------
# One-call facade
# ---------------------------------------------------------------------------


def convert_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    mode: str = "cpu",
    out_format: str = DEFAULT_OUTPUT_FORMAT,
    level: int | None = None,
    encoder: str | None = None,
    png_strategy: str = DEFAULT_PNG_STRATEGY,
    workers: int | None = None,
    device: str = "auto",
    batch_size: int = 1,
    skip_existing: bool = False,
    limit: int | None = None,
    log_path: str | Path | None = None,
    on_scanned: Callable[[ScanResult], None] | None = None,
    progress_cb: ProgressCB | None = None,
    cancel: threading.Event | None = None,
    stats: RunStats | None = None,
) -> RunStats:
    """Scan, mirror the folder tree, convert, and optionally log the run.

    ``workers`` is the pipeline width (see :func:`default_workers`); ``1`` means
    no pool and no threads at all.  ``level`` is the format's compression control
    (PNG 0-9, JPEG quality 1-100), ignored by formats that have none.

    ``encoder`` picks who writes the file; ``png_strategy`` is OpenCV-only and
    PNG-only.  Reading and format sniffing are always Pillow's, see
    :data:`ENCODERS`.
    """
    if out_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unknown output format {out_format!r}; "
                         f"expected one of {sorted(OUTPUT_FORMATS)}")

    scan = scan_input(input_dir, output_dir,
                      out_format=out_format, skip_existing=skip_existing)
    if limit is not None:
        scan.tasks = scan.tasks[:limit]
    if on_scanned is not None:
        on_scanned(scan)

    # Recreate the *entire* nested structure first, empty folders included.
    mirror_tree(output_dir, scan.directories)

    if mode == "gpu":
        result = run_gpu(scan.tasks, out_format=out_format, level=level,
                         encoder=encoder, png_strategy=png_strategy,
                         device=device, batch_size=batch_size, workers=workers,
                         progress_cb=progress_cb, cancel=cancel, stats=stats)
    else:
        result = run_cpu(scan.tasks, out_format=out_format, level=level,
                         encoder=encoder, png_strategy=png_strategy,
                         workers=workers, progress_cb=progress_cb,
                         cancel=cancel, stats=stats)

    # Only an aborted or partly failed run can leave staging files behind, so
    # skip the extra tree walk on the happy path.
    if result.cancelled or result.failed:
        sweep_temp_files(output_dir)

    if log_path:
        # A failed log must never fail the conversion -- but it must not be
        # silent either, or the UI reports a file that was never written.
        record = None
        try:
            record = build_run_record(result, input_dir, output_dir, scan)
            append_run_log(log_path, record)
        except Exception as exc:
            result.log_error = f"{type(exc).__name__}: {exc}"
        # Reported separately: the run log can land while its failure list does
        # not, and saying "log not written" for that would be a lie.
        if record is not None and result.errors:
            try:
                result.failures_path = str(append_failure_log(
                    log_path, record["run_id"], record["timestamp"],
                    result.errors))
            except Exception as exc:
                result.failures_error = f"{type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def human_bytes(n: float) -> str:
    """Format a byte count for the UI/report."""
    if abs(n) < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
    return f"{n / 1024:.1f} TB"


def human_time(seconds: float) -> str:
    """``73.4`` -> ``01:13``; ``4000`` -> ``1:06:40``."""
    if seconds < 0 or seconds != seconds:  # negative or NaN
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def configure_start_method() -> str:
    """Pick a multiprocessing start method that is safe next to a GUI.

    ``fork`` copies the parent including its threads' held locks, which is a
    classic source of hangs when the parent is a Tk app, so prefer
    ``forkserver`` on POSIX and ``spawn`` on Windows.  Returns the method in
    force; safe to call more than once.
    """
    current = mp.get_start_method(allow_none=True)
    if current is not None:
        return current
    for method in ("forkserver", "spawn"):
        try:
            mp.set_start_method(method)
            return method
        except (ValueError, RuntimeError):
            continue
    return mp.get_start_method()
