"""
app.py -- desktop GUI for batch-converting image datasets to 8-bit grayscale.

    python app.py

The engine lives in ``gray_core``, the widget set in ``ui_kit``; this file is
the wiring in between -- it collects the settings, starts a run, and draws the
progress.

Three rules it obeys, each of which is a way the app would otherwise break:

1. **The heavy loop never runs on the Tk thread.**  ``convert_dataset`` goes on
   a ``threading.Thread``; run inline, the window would stop redrawing and the
   OS would mark it "Not Responding".
2. **Nothing but the Tk thread touches a widget.**  The worker only mutates a
   shared :class:`gray_core.RunStats` and pushes log lines onto a
   ``queue.Queue``; a ``root.after`` poller reads those and does the drawing.
   Calling Tk from another thread crashes intermittently.
3. **Everything that starts the app sits behind ``if __name__ == "__main__"``**
   with ``freeze_support()``.  Without it a spawned child re-importing this
   module would build a second GUI -- and that child would spawn more, forever.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import traceback
from pathlib import Path

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox

from . import gray_core as gc
from . import ui_kit as ui


class ConverterApp(tk.Tk):
    POLL_MS = 100          # how often the Tk thread refreshes progress
    LOG_DRAIN = 200        # max log lines folded into one refresh

    def __init__(self) -> None:
        super().__init__()
        self.title("Image → Grayscale")
        self.configure(bg=ui.BG)
        self.geometry("940x900")
        self.minsize(880, 700)

        # -- shared state between the Tk thread and the worker thread ------
        self.stats: gc.RunStats | None = None
        self.cancel_event = threading.Event()
        self.log_q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.running = False

        # -- Tk variables --------------------------------------------------
        self.var_input = tk.StringVar()
        self.var_output = tk.StringVar()
        self.var_mode = tk.StringVar(value="cpu")
        # Pipeline width, same unit in both modes -- see gc.default_workers().
        self.var_workers = tk.IntVar(value=gc.default_workers())
        # 1, not 8: larger batches measured monotonically worse on real CUDA.
        self.var_batch = tk.IntVar(value=1)
        # One variable per format: PNG's 0-9 and JPEG's 1-100 are different
        # scales, so switching formats must not carry a value across.
        self.var_level = {k: tk.IntVar(value=f.level.default)
                          for k, f in gc.OUTPUT_FORMATS.items() if f.level}
        self.var_format = tk.StringVar(value=gc.DEFAULT_OUTPUT_FORMAT)
        # Encoder and PNG strategy are comparison controls, like GPU mode:
        # kept so a run can be measured against another, not because one is
        # known to be right.  Defaults are the values the app always had.
        self.var_encoder = tk.StringVar(value=gc.DEFAULT_ENCODER)
        self.var_strategy = tk.StringVar(value=gc.DEFAULT_PNG_STRATEGY)
        self.var_skip_existing = tk.BooleanVar(value=False)
        self.var_log = tk.BooleanVar(value=False)
        self.var_log_path = tk.StringVar()

        self.gpu_device, self.gpu_reason = gc.detect_gpu()
        self.cv_ok, self.cv_reason = gc.detect_encoder()

        self._init_fonts()
        self._build_ui()
        self._sync_mode_widgets()
        self._sync_format_note()
        self._sync_log_row()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(self.POLL_MS, self._tick)

    # ------------------------------------------------------------- styling
    def _init_fonts(self) -> None:
        """Resolve one UI family and one monospace family, then build the scale."""
        fam = ui.pick_font(self, ui.UI_FAMILIES)
        mono = ui.pick_font(self, ui.MONO_FAMILIES, fallback="Courier")
        self.font_family, self.mono_family = fam, mono
        self.fonts = {
            "h1": tkfont.Font(family=fam, size=18, weight="bold"),
            "h2": tkfont.Font(family=fam, size=13, weight="bold"),
            "num": tkfont.Font(family=fam, size=26, weight="bold"),
            "stat": tkfont.Font(family=fam, size=15, weight="bold"),
            "body": tkfont.Font(family=fam, size=10),
            "body_b": tkfont.Font(family=fam, size=10, weight="bold"),
            "small": tkfont.Font(family=fam, size=9),
            "section": tkfont.Font(family=fam, size=8, weight="bold"),
            "mono": tkfont.Font(family=mono, size=9),
        }
        self.fonts_ok = ui.has_scalable_fonts(self)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        page = tk.Frame(self, bg=ui.BG)
        page.pack(fill="both", expand=True, padx=22, pady=(14, 14))

        self._build_header(page)
        self._build_folders(page)
        self._build_processing(page)
        self._build_footer(page)      # claims its strip at the bottom first…
        self._build_progress(page)    # …so this card gets whatever is left

    # ---- header ----------------------------------------------------------
    def _build_header(self, page) -> None:
        head = tk.Frame(page, bg=ui.BG)
        head.pack(fill="x", pady=(0, 12))

        left = tk.Frame(head, bg=ui.BG)
        left.pack(side="left")
        tk.Label(left, text="Image → Grayscale", bg=ui.BG, fg=ui.INK,
                 font=self.fonts["h1"], anchor="w").pack(anchor="w")
        tk.Label(left, text="Batch 8-bit conversion · any image format in · "
                            "mirrors the tree · keeps paired .json annotations",
                 bg=ui.BG, fg=ui.MUTED, font=self.fonts["small"],
                 anchor="w").pack(anchor="w", pady=(3, 0))

        self.badge = ui.Pill(head, self.fonts["small"], bg=ui.BG)
        self.badge.pack(side="right", pady=(6, 0))

    # ---- folders ---------------------------------------------------------
    def _build_folders(self, page) -> None:
        card = ui.Card(page)
        card.pack(fill="x", pady=(0, 10))
        body = card.body
        ui.section(body, "Folders", self.fonts["section"]).pack(anchor="w", pady=(0, 9))

        grid = tk.Frame(body, bg=ui.CARD)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        self._folder_row(grid, 0, "Input", self.var_input, self._browse_input)
        self._folder_row(grid, 1, "Output", self.var_output, self._browse_output)

        self.lbl_scan = tk.Label(
            body, text="Choose a source folder — every image underneath it will "
                       "be converted, and the tree recreated in the destination.",
            bg=ui.CARD, fg=ui.MUTED, font=self.fonts["small"], anchor="w",
            justify="left")
        self.lbl_scan.pack(anchor="w", pady=(9, 0))

    def _folder_row(self, parent, row, label, var, browse) -> None:
        tk.Label(parent, text=label, bg=ui.CARD, fg=ui.BODY, font=self.fonts["body"],
                 width=7, anchor="w").grid(row=row, column=0, sticky="w",
                                           pady=(0, 10) if row == 0 else 0)
        ui.Field(parent, var, self.fonts["body"]).grid(
            row=row, column=1, sticky="ew", padx=(2, 10),
            pady=(0, 10) if row == 0 else 0)
        ui.Button(parent, "Browse", browse, kind="secondary",
                  font=self.fonts["body"], height=32, pad_x=16).grid(
            row=row, column=2, pady=(0, 10) if row == 0 else 0)

    # ---- processing + options -------------------------------------------
    def _build_processing(self, page) -> None:
        card = ui.Card(page)
        card.pack(fill="x", pady=(0, 10))
        body = card.body
        ui.section(body, "Processing mode", self.fonts["section"]).pack(
            anchor="w", pady=(0, 12))

        top = tk.Frame(body, bg=ui.CARD)
        top.pack(fill="x")
        self.seg_mode = ui.Segmented(
            top, [("cpu", "CPU  ·  multiprocessing"), ("gpu", "GPU  ·  CUDA / MPS")],
            self.var_mode, command=self._sync_mode_widgets,
            font=self.fonts["body"], font_active=self.fonts["body_b"], height=38)
        self.seg_mode.pack(side="left")
        self.lbl_device = tk.Label(top, bg=ui.CARD, fg=ui.MUTED,
                                   font=self.fonts["small"], anchor="w",
                                   justify="left", wraplength=380)
        self.lbl_device.pack(side="left", padx=(16, 0))

        ui.divider(body, pady=(10, 9))

        fmt_row = tk.Frame(body, bg=ui.CARD)
        fmt_row.pack(fill="x")
        self._knob_label(fmt_row, "Output format")
        self.seg_format = ui.Segmented(
            fmt_row,
            [(k, gc.OUTPUT_FORMATS[k].label) for k in gc.OUTPUT_FORMATS],
            self.var_format, command=self._sync_format_note,
            font=self.fonts["small"], font_active=self.fonts["small"],
            height=32, seg_width=72)
        self.seg_format.pack(side="left", padx=(10, 14))
        self.lbl_format = tk.Label(fmt_row, bg=ui.CARD, fg=ui.FAINT,
                                   font=self.fonts["small"], anchor="w")
        self.lbl_format.pack(side="left")

        enc_row = tk.Frame(body, bg=ui.CARD)
        enc_row.pack(fill="x", pady=(7, 0))
        self._knob_label(enc_row, "Encoder")
        self.seg_encoder = ui.Segmented(
            enc_row, [("pillow", "Pillow"), ("opencv", "OpenCV")],
            self.var_encoder, command=self._sync_format_note,
            font=self.fonts["small"], font_active=self.fonts["small"],
            height=32, seg_width=84)
        self.seg_encoder.pack(side="left", padx=(10, 14))
        self.lbl_strategy = self._knob_label(enc_row, "PNG strategy")
        self.seg_strategy = ui.Segmented(
            enc_row, [(k, k.capitalize()) for k in gc.PNG_STRATEGIES],
            self.var_strategy, command=self._sync_format_note,
            font=self.fonts["small"], font_active=self.fonts["small"],
            height=32, seg_width=72)
        self.seg_strategy.pack(side="left", padx=(10, 14))
        self.lbl_encoder_hint = tk.Label(enc_row, bg=ui.CARD, fg=ui.FAINT,
                                         font=self.fonts["small"], anchor="w")
        self.lbl_encoder_hint.pack(side="left")

        knobs = tk.Frame(body, bg=ui.CARD)
        knobs.pack(fill="x", pady=(7, 0))
        # Compression sits with the other steppers rather than on a row of its
        # own; its explanation shares the encoder row's hint, where the rest of
        # the write settings already live.
        self.lbl_level = self._knob_label(knobs, "Compression")
        self.step_level = ui.Stepper(knobs, next(iter(self.var_level.values())),
                                     0, 100, font=self.fonts["body_b"])
        self.step_level.pack(side="left", padx=(10, 26))
        self.lbl_workers = self._knob_label(knobs, "Workers")
        self.step_workers = ui.Stepper(knobs, self.var_workers, 1,
                                       max(64, (os.cpu_count() or 8) * 4),
                                       font=self.fonts["body_b"],
                                       command=self._sync_mode_widgets)
        self.step_workers.pack(side="left", padx=(10, 26))
        self.lbl_batch = self._knob_label(knobs, "Batch size")
        self.step_batch = ui.Stepper(knobs, self.var_batch, 1, 256,
                                     font=self.fonts["body_b"])
        self.step_batch.pack(side="left", padx=(10, 26))
        self.lbl_hint = tk.Label(knobs, bg=ui.CARD, fg=ui.FAINT,
                                 font=self.fonts["small"], anchor="w")
        self.lbl_hint.pack(side="left")

        ui.Check(body, "Skip images already present in the output — lets an "
                       "interrupted run resume where it stopped",
                 self.var_skip_existing,
                 font=self.fonts["body"]).pack(anchor="w", pady=(11, 0))

        log_row = tk.Frame(body, bg=ui.CARD)
        log_row.pack(fill="x", pady=(7, 0))
        ui.Check(log_row, "Log this run to", self.var_log,
                 command=self._sync_log_row,
                 font=self.fonts["body"]).pack(side="left")
        self.btn_log = ui.Button(log_row, "…", self._browse_log, kind="secondary",
                                 font=self.fonts["small"], height=30, pad_x=12)
        self.btn_log.pack(side="right")
        self.field_log = ui.Field(log_row, self.var_log_path, self.fonts["small"],
                                  pad=(9, 6))
        self.field_log.pack(side="left", fill="x", expand=True, padx=(10, 8))

    def _sync_format_note(self) -> None:
        """Show what the chosen format costs, and re-point the level control."""
        key = self.var_format.get()
        fmt = gc.OUTPUT_FORMATS[key]

        knob = fmt.level
        # rle constrains zlib's match distance to 1, so the level has nothing
        # left to search: levels 1 and 9 emit identical bytes.  A live stepper
        # that changes nothing is worse than a greyed one.
        inert = (key == "png" and self.var_encoder.get() == "opencv"
                 and self.var_strategy.get() == "rle")
        self.step_level.set_enabled(knob is not None and not inert)
        self.lbl_level.configure(fg=ui.MUTED if knob and not inert else ui.FAINT)
        if knob is None:
            note = ("no compression to tune" if key == "bmp"
                    else "LZW; Pillow exposes no level")
        else:
            self.step_level.set_bounds(knob.lo, knob.hi)
            self.step_level.set_variable(self.var_level[key])
            note = ("levels 1-9 identical under rle" if inert else knob.hint)
        # The format row is the one with width to spare, so the compression
        # note rides along with the format note rather than crowding its own row.
        self.lbl_format.configure(text=f"{fmt.note}  ·  {note}",
                                  fg=ui.FAINT if fmt.lossless else ui.WARN)
        self._sync_encoder_row(key)

    def _sync_encoder_row(self, key: str) -> None:
        """Reading is always Pillow's; only the write side is switchable.

        The PNG strategy exists only in OpenCV's encoder, so it is live only for
        that combination -- and it is worth saying that ``rle`` makes the
        compression level irrelevant, rather than letting a stepper that does
        nothing look broken.
        """
        self.seg_encoder.set_disabled(() if self.cv_ok else ("opencv",))
        if not self.cv_ok and self.var_encoder.get() == "opencv":
            self.var_encoder.set("pillow")
        opencv_png = self.var_encoder.get() == "opencv" and key == "png"
        self.seg_strategy.set_disabled(() if opencv_png else gc.PNG_STRATEGIES)
        self.lbl_strategy.configure(fg=ui.MUTED if opencv_png else ui.FAINT)
        if not self.cv_ok:
            hint = self.cv_reason
        elif opencv_png and self.var_strategy.get() == "rle":
            hint = "rle: smaller AND faster here"
        else:
            hint = "reads always stay Pillow's"
        self.lbl_encoder_hint.configure(text=hint)

    def _level_for(self, key: str) -> int | None:
        """The compression level to run with, or ``None`` for BMP and TIFF."""
        var = self.var_level.get(key)
        return None if var is None else var.get()

    def _sync_log_row(self) -> None:
        on = self.var_log.get()
        self.field_log.entry.configure(state="normal" if on else "disabled")
        self.btn_log.set_enabled(on)
        if on and not self.var_log_path.get():
            self._default_log_path()

    def _default_log_path(self) -> None:
        out = self.var_output.get().strip()
        if out:
            self.var_log_path.set(str(Path(out) / "bmp2gray_runs.csv"))

    def _browse_log(self) -> None:
        chosen = filedialog.asksaveasfilename(
            title="Run log file", defaultextension=".csv",
            initialfile="bmp2gray_runs.csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if chosen:
            self.var_log_path.set(chosen)

    def _knob_label(self, parent, text) -> tk.Label:
        lbl = tk.Label(parent, text=text, bg=ui.CARD, fg=ui.MUTED,
                       font=self.fonts["small"])
        lbl.pack(side="left")
        return lbl

    # ---- progress --------------------------------------------------------
    def _build_progress(self, page) -> None:
        card = ui.Card(page)
        card.pack(fill="both", expand=True)
        body = card.body

        head = tk.Frame(body, bg=ui.CARD)
        head.pack(fill="x")
        self.lbl_pct = tk.Label(head, text="0%", bg=ui.CARD, fg=ui.INK,
                                font=self.fonts["num"], anchor="w")
        self.lbl_pct.pack(side="left")
        self.lbl_status = tk.Label(head, text="Idle", bg=ui.CARD, fg=ui.MUTED,
                                   font=self.fonts["body"], anchor="e")
        self.lbl_status.pack(side="right", pady=(12, 0))

        self.bar = ui.Bar(body)
        self.bar.pack(fill="x", pady=(8, 11))

        tiles = tk.Frame(body, bg=ui.CARD)
        tiles.pack(fill="x", pady=(0, 10))
        self.tiles = {}
        for i, (key, caption) in enumerate((("images", "Converted"),
                                            ("ann", "Annotations"),
                                            ("failed", "Errors"),
                                            ("size", "Written"))):
            tiles.columnconfigure(i, weight=1, uniform="tile")
            tile = ui.Stat(tiles, caption, self.fonts["stat"], self.fonts["small"])
            tile.grid(row=0, column=i, sticky="w")
            self.tiles[key] = tile

        shell = tk.Frame(body, bg=ui.FIELD, highlightthickness=1,
                         highlightbackground=ui.BORDER, highlightcolor=ui.BORDER)
        shell.pack(fill="both", expand=True)
        self.log = tk.Text(shell, height=6, wrap="none", bg=ui.FIELD, fg=ui.BODY,
                           relief="flat", font=self.fonts["mono"], padx=12, pady=9,
                           insertbackground=ui.BODY, selectbackground=ui.ACCENT_SOFT,
                           selectforeground=ui.INK, highlightthickness=0, bd=0)
        scroll = ui.Scrollbar(shell, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set, state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y", padx=(0, 3), pady=3)
        self.log.tag_configure("err", foreground=ui.ERR)
        self.log.tag_configure("ok", foreground=ui.OK)
        self.log.tag_configure("info", foreground=ui.ACCENT)
        self.log.tag_configure("dim", foreground="#8b939e")

    # ---- footer ----------------------------------------------------------
    def _build_footer(self, page) -> None:
        bar = tk.Frame(page, bg=ui.BG)
        bar.pack(side="bottom", fill="x", pady=(10, 0))
        self.btn_scan = ui.Button(bar, "Scan input", self._on_scan, kind="ghost",
                                  font=self.fonts["body"], bg=ui.BG, height=38)
        self.btn_scan.pack(side="left")
        self.btn_start = ui.Button(bar, "Start Conversion", self._on_start,
                                   kind="primary", font=self.fonts["body_b"],
                                   bg=ui.BG, height=38, pad_x=22)
        self.btn_start.pack(side="right")
        self.btn_cancel = ui.Button(bar, "Cancel", self._on_cancel, kind="secondary",
                                    font=self.fonts["body"], bg=ui.BG, height=38)
        self.btn_cancel.pack(side="right", padx=(0, 10))
        self.btn_cancel.set_enabled(False)

    # ------------------------------------------------------------- helpers
    def _log(self, msg: str, tag: str = "") -> None:
        """Append a line to the log pane.  Tk thread only."""
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _sync_mode_widgets(self) -> None:
        """Match every control to the selected mode and the detected hardware."""
        gpu_ok = self.gpu_device is not None

        if self.gpu_device is not None:
            self.badge.set(self.gpu_reason, ui.OK, "#e7f6ee")
            self.lbl_device.configure(text="Accelerator ready — images are batched "
                                           "and converted on the device.")
        else:
            self.badge.set("CPU only", ui.MUTED, "#ecedf0")
            self.lbl_device.configure(text=self.gpu_reason)

        self.seg_mode.set_disabled(() if gpu_ok else ("gpu",))
        if not gpu_ok and self.var_mode.get() == "gpu":  # never leave GPU selected
            self.var_mode.set("cpu")
            self.seg_mode._draw()

        cpu_mode = self.var_mode.get() == "cpu"
        self.step_batch.set_enabled(not cpu_mode)
        self.lbl_batch.configure(fg=ui.FAINT if cpu_mode else ui.MUTED)
        # The width is the same in both modes, its cost is not -- so spell the
        # cost out rather than leave "why twice the threads?" to be asked.
        n = self.var_workers.get()
        if cpu_mode:
            hint = f"{n} wide · {n} process(es)"
            # BMP flattens at the physical core count; the compressing formats
            # were still climbing at the logical one.  Say so instead of moving
            # the stepper for them -- see README, "Workers".
            if (self.var_format.get() != "bmp"
                    and n <= gc.physical_cores() < (os.cpu_count() or n)):
                hint += f"  ·  compressing formats kept scaling to {os.cpu_count()} here"
        elif n == 1:
            hint = "1 wide · serial, no threads"
        else:
            hint = f"{n} wide · {n}+{n} = {2 * n} threads"
        self.lbl_hint.configure(text=hint)

    def _set_running(self, running: bool) -> None:
        self.running = running
        for button in (self.btn_start, self.btn_scan):
            button.set_enabled(not running)
        self.btn_cancel.set_enabled(running)

    # ------------------------------------------------------------ browsing
    def _browse_input(self) -> None:
        chosen = filedialog.askdirectory(title="Select the input dataset folder")
        if chosen:
            self.var_input.set(chosen)
            if not self.var_output.get():
                # Offer a sensible sibling destination.
                self.var_output.set(str(Path(chosen).parent /
                                        (Path(chosen).name + "_grayscale")))

    def _browse_output(self) -> None:
        chosen = filedialog.askdirectory(title="Select the output folder")
        if chosen:
            self.var_output.set(chosen)
            if self.var_log.get() and not self.var_log_path.get():
                self._default_log_path()

    # ---------------------------------------------------------------- scan
    def _validate(self) -> tuple[Path, Path] | None:
        in_dir = self.var_input.get().strip()
        out_dir = self.var_output.get().strip()
        if not in_dir or not Path(in_dir).is_dir():
            messagebox.showerror("Input folder", "Pick an existing input folder first.")
            return None
        if not out_dir:
            messagebox.showerror("Output folder", "Pick an output folder first.")
            return None
        src, dst = Path(in_dir).resolve(), Path(out_dir).expanduser().resolve()
        if src == dst:
            messagebox.showerror("Folders", "Input and output folders must differ — "
                                            "this tool never overwrites the originals.")
            return None
        if src in dst.parents:
            if not messagebox.askyesno(
                    "Output is inside the input",
                    "The output folder sits inside the input folder.\n\n"
                    "It will be excluded from the scan so it can't convert its own "
                    "output, but a separate destination is cleaner.\n\n"
                    "Continue anyway?"):
                return None
        return src, dst

    def _on_scan(self) -> None:
        paths = self._validate()
        if not paths:
            return
        src, dst = paths
        self.lbl_status.configure(text="Scanning…")
        self.update_idletasks()
        try:
            scan = gc.scan_input(src, dst, out_format=self.var_format.get(),
                                 skip_existing=self.var_skip_existing.get())
        except Exception as exc:
            messagebox.showerror("Scan failed", str(exc))
            self.lbl_status.configure(text="Scan failed")
            return

        images = len(scan.tasks)
        paired = sum(1 for t in scan.tasks if t.ann_src is not None)
        kinds = ", ".join(f"{n:,}\u00d7{ext}" for ext, n in
                          sorted(scan.suffixes_seen.items(), key=lambda kv: -kv[1]))
        summary = (f"{images:,} images ({kinds}) · {paired:,} paired .json · "
                   f"{len(scan.directories):,} sub-folders")
        if scan.orphan_annotations:
            summary += f" · {len(scan.orphan_annotations):,} unpaired .json"
        if scan.skipped_files:
            summary += f" · {len(scan.skipped_files):,} non-image files ignored"
        if scan.shadowed:
            summary += (f" · {len(scan.shadowed):,} shadowed by a same-named "
                        f"source")
        if scan.already_done:
            summary += f" · {scan.already_done:,} already converted"
        self.lbl_scan.configure(text=summary, fg=ui.BODY)

        # Converting an already-compressed source to BMP shrinks the pixels but
        # grows the file. Worth saying before the run, not after it.
        compressed = {e for e in scan.suffixes_seen
                      if e not in (".bmp", ".ppm", ".pgm", ".pnm", ".tga")}
        for src_rel, out_name in scan.shadowed[:10]:
            self._log(f"shadow {src_rel} -> not converted; another source in that "
                      f"folder already produces {out_name}", "dim")
        if compressed and self.var_format.get() == "bmp":
            self._log(f"note   {', '.join(sorted(compressed))} sources are already "
                      f"compressed - BMP output will be LARGER than the input", "dim")
        self.lbl_status.configure(text=f"Ready — {images:,} image(s) queued")
        self._log("scan  " + summary, "info")

    # --------------------------------------------------------------- start
    def _on_start(self) -> None:
        if self.running:
            return
        paths = self._validate()
        if not paths:
            return
        src, dst = paths

        mode = self.var_mode.get()
        if mode == "gpu" and self.gpu_device is None:
            messagebox.showerror("GPU unavailable", self.gpu_reason)
            return

        if dst.exists() and any(dst.iterdir()):
            if not messagebox.askyesno(
                    "Output folder is not empty",
                    f"{dst}\n\nalready contains files. Existing files with the same "
                    "names will be overwritten.\n\nContinue?"):
                return

        self.cancel_event = threading.Event()
        self.stats = gc.RunStats()
        self.bar.set(0, ui.ACCENT)
        self.lbl_pct.configure(text="0%", fg=ui.INK)
        for tile in self.tiles.values():
            tile.set("0")
        self.lbl_status.configure(text="Scanning…")
        self._set_running(True)

        log_path = self.var_log_path.get().strip() if self.var_log.get() else None
        kwargs = dict(
            mode=mode,
            out_format=self.var_format.get(),
            level=self._level_for(self.var_format.get()),
            encoder=self.var_encoder.get(),
            png_strategy=self.var_strategy.get(),
            workers=self.var_workers.get(),
            batch_size=self.var_batch.get(),
            skip_existing=self.var_skip_existing.get(),
            log_path=log_path or None,
        )
        n = kwargs["workers"]
        cost = (f"{n} process(es)" if mode == "cpu" else
                "serial, no threads" if n == 1 else f"{2 * n} threads")
        detail = f"width={n} ({cost})"
        if mode == "gpu":
            detail += f", batch={kwargs['batch_size']}, device={self.gpu_device}"
        # Ask the engine what it resolved rather than reading the widgets back:
        # a control that does not apply to the chosen format is blanked there,
        # and only that copy is the one the encoder will actually see.
        fmt = gc.resolve_format(kwargs["out_format"], kwargs["level"],
                                encoder=kwargs["encoder"],
                                png_strategy=kwargs["png_strategy"])
        detail += f", format={fmt.key}"
        if fmt.effective_level is not None:
            detail += f" (level {fmt.effective_level})"
        detail += f", encoder={fmt.encoder}"
        if fmt.png_strategy:
            detail += f"/{fmt.png_strategy}"
        self._log(f"start  mode={mode} · {detail}", "info")

        self.worker = threading.Thread(target=self._run_worker, args=(src, dst, kwargs),
                                       name="conversion", daemon=True)
        self.worker.start()

    def _run_worker(self, src: Path, dst: Path, kwargs: dict) -> None:
        """Runs OFF the Tk thread.  Must not touch a single widget."""
        try:
            gc.convert_dataset(
                src, dst,
                on_scanned=lambda scan: self.log_q.put(
                    ("dim", f"queued  {len(scan.tasks):,} task(s) across "
                            f"{len(scan.directories):,} sub-folder(s)")),
                progress_cb=self._progress_cb,
                cancel=self.cancel_event,
                stats=self.stats,
                **kwargs,
            )
        except Exception as exc:
            self.log_q.put(("err", f"fatal  {type(exc).__name__}: {exc}"))
            for line in traceback.format_exc().splitlines()[-4:]:
                self.log_q.put(("dim", "       " + line))
        finally:
            self.log_q.put(("__done__", ""))

    def _progress_cb(self, result: gc.FileResult, stats: gc.RunStats) -> None:
        """Called once per file from a worker thread.

        Only failures are queued — pushing 1000+ success lines through a queue
        and into a Text widget would cost more than the conversion itself.  The
        counters are read straight off ``stats`` by the Tk-side poller.
        """
        if not result.ok:
            self.log_q.put(("err", f"error  {Path(result.src).name}: {result.error}"))

    # ------------------------------------------------------- Tk-side poller
    def _tick(self) -> None:
        """Runs every POLL_MS on the Tk thread: drains the log queue, redraws."""
        finished = False
        for _ in range(self.LOG_DRAIN):  # bounded drain keeps the UI responsive
            try:
                tag, msg = self.log_q.get_nowait()
            except queue.Empty:
                break
            if tag == "__done__":
                finished = True
            else:
                self._log(msg, tag)

        st = self.stats
        if st is not None and self.running:
            pct = 100.0 * st.done / max(st.total, 1)
            self.bar.set(pct)
            self.lbl_pct.configure(text=f"{pct:.0f}%")
            eta = (st.total - st.done) / st.rate if st.rate > 0 else float("nan")
            self.lbl_status.configure(
                text=f"{st.done:,} / {st.total:,}   ·   {st.rate:.1f} img/s"
                     f"   ·   ETA {gc.human_time(eta)}")
            self._refresh_tiles(st)

        if finished:
            self._on_finished()

        self.after(self.POLL_MS, self._tick)

    def _refresh_tiles(self, st: gc.RunStats) -> None:
        self.tiles["images"].set(f"{st.ok:,}")
        self.tiles["ann"].set(f"{st.ann_copied:,}")
        self.tiles["failed"].set(f"{st.failed:,}", ui.ERR if st.failed else ui.INK)
        self.tiles["size"].set(gc.human_bytes(st.bytes_out))

    def _on_finished(self) -> None:
        self._set_running(False)
        st = self.stats
        if st is None:
            return
        self._refresh_tiles(st)
        saved = st.bytes_in - st.bytes_out
        pct_saved = (100.0 * saved / st.bytes_in) if st.bytes_in else 0.0
        cancelled = st.cancelled

        if not cancelled:
            self.bar.set(100, ui.ERR if st.failed else ui.OK)
            self.lbl_pct.configure(text="100%", fg=ui.ERR if st.failed else ui.OK)
        else:
            self.bar.set(100.0 * st.done / max(st.total, 1), ui.WARN)

        verdict = "Cancelled" if cancelled else "Done"
        self.lbl_status.configure(
            text=f"{verdict} — {st.ok:,} image(s) in {gc.human_time(st.elapsed)}"
                 f"   ·   {st.rate:.1f} img/s")
        self._log(
            f"{verdict.lower():<6} {st.ok:,} converted · {st.ann_copied:,} .json"
            + (f" ({st.ann_rewritten:,} imagePath rewritten)" if st.ann_rewritten else "")
            + f" · {st.failed:,} failed  |  "
            f"{gc.human_time(st.elapsed)} @ {st.rate:.1f} img/s  |  "
            f"{gc.human_bytes(st.bytes_in)} → {gc.human_bytes(st.bytes_out)} "
            f"({pct_saved:.1f}% smaller)",
            "err" if st.failed else ("dim" if cancelled else "ok"))
        if st.estimated_s:
            self._log(f"       predicted {gc.human_time(st.estimated_s)} after "
                      f"{gc.RunStats.ESTIMATE_AFTER} images, actual "
                      f"{gc.human_time(st.elapsed)}", "dim")
        path = self.var_log_path.get().strip()
        if self.var_log.get() and path:
            if st.log_error:
                self._log(f"log     NOT written to {path} — {st.log_error}", "err")
            else:
                self._log(f"logged  {path}", "dim")

    # -------------------------------------------------------------- cancel
    def _on_cancel(self) -> None:
        if not self.running:
            return
        self.cancel_event.set()
        self.btn_cancel.set_enabled(False)
        self.lbl_status.configure(text="Cancelling — finishing files in flight…")
        self._log("cancel requested", "dim")

    def _on_close(self) -> None:
        if self.running:
            if not messagebox.askyesno("Quit", "A conversion is running. "
                                               "Stop it and quit?"):
                return
            self.cancel_event.set()
        self.destroy()


def main() -> None:
    # THE multiprocessing + GUI gotcha.  ``freeze_support`` is a no-op on a
    # normal CPython run but is essential once the app is frozen with
    # PyInstaller/cx_Freeze on Windows: without it every worker process would
    # re-run this file from the top and open its own window, recursively.
    # It lives inside ``main`` rather than under ``if __name__ == "__main__"``
    # because the installed ``rgb2gray`` console script calls ``main`` directly
    # and never executes this module's ``__main__`` block.
    mp.freeze_support()

    # Pick a start method that is safe to use alongside a Tk main loop.
    method = gc.configure_start_method()
    app = ConverterApp()
    app._log(f"ready  {os.cpu_count()} logical CPU(s) · start method: {method} · "
             f"font: {app.font_family}", "dim")
    if not app.fonts_ok:
        app._log("note   this Python's Tk was built without Xft, so text is not "
                 "anti-aliased — see README, 'Fonts look wrong'", "dim")
    app.mainloop()


if __name__ == "__main__":
    main()
