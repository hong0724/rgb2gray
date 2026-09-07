"""
ui_kit.py -- a tiny flat-design widget set drawn on Tk canvases.

Stock ttk widgets carry a lot of Motif DNA: sunken entry borders, 3-D spinbox
arrows, chunky check indicators.  Rather than fight the theme engine, the
handful of controls this app needs are drawn directly on ``tk.Canvas``.  Pure
tkinter, no extra dependency, identical on Linux, macOS and Windows.

One trap worth knowing: Tk silently falls back to a bitmap font when a family is
missing, and some CPython builds ship a Tk compiled **without** Xft, which has
no anti-aliasing and no access to system fonts at all.  :func:`pick_font` reports
what it actually got so the app can explain why the text looks the way it does.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

BG = "#f4f5f7"          # page
CARD = "#ffffff"        # card surface
FIELD = "#fbfcfd"       # input surface
BORDER = "#e3e6ea"      # hairline
BORDER_D = "#cfd5dc"    # stronger hairline (inputs, check outlines)
DIVIDER = "#eef0f3"

INK = "#11161c"         # headings
BODY = "#3a424c"        # normal text
MUTED = "#6b7280"       # secondary text
FAINT = "#a3aab4"       # disabled

ACCENT = "#2f6fed"
ACCENT_D = "#2059d0"
ACCENT_SOFT = "#eaf1fe"
ACCENT_DIM = "#b9cdf7"  # disabled primary

OK = "#0f8a4e"
ERR = "#d03434"
WARN = "#b7791f"

TRACK = "#e8eaee"
SEG_TRACK = "#f0f1f4"
HOVER = "#f5f6f8"

UI_FAMILIES = (
    "Inter", "SF Pro Text", "Segoe UI Variable Text", "Segoe UI", "Noto Sans",
    "Cantarell", "Ubuntu", "Roboto", "Open Sans", "DejaVu Sans",
    "Liberation Sans", "Helvetica",
)
MONO_FAMILIES = (
    "JetBrains Mono", "JetBrainsMono Nerd Font", "SF Mono", "Cascadia Mono",
    "Consolas", "Noto Sans Mono", "Ubuntu Mono", "DejaVu Sans Mono",
    "Liberation Mono", "Courier",
)


def pick_font(root: tk.Misc, families=UI_FAMILIES, fallback: str = "Helvetica") -> str:
    """First installed family from ``families`` (case-insensitive), else fallback."""
    have = {f.lower(): f for f in tkfont.families(root)}
    for name in families:
        if name.lower() in have:
            return have[name.lower()]
    return fallback


def has_scalable_fonts(root: tk.Misc) -> bool:
    """False on a Tk built without Xft: it reports a few dozen bitmap families
    instead of the several hundred fontconfig knows about."""
    return len(tkfont.families(root)) > 80


# ---------------------------------------------------------------------------
# Canvas primitive
# ---------------------------------------------------------------------------


# NOTE: never store geometry on ``self._w`` in a widget subclass -- tkinter uses
# that attribute for the Tcl widget path. These classes use ``_cw`` / ``_ch``.


def round_rect(cv: tk.Canvas, x1, y1, x2, y2, r, **kw):
    """Rounded rectangle: Tk has no such item, so duplicate the corner points
    and let the spline smoother round them.  One item, no arcs to line up."""
    r = max(0, min(r, abs(x2 - x1) / 2, abs(y2 - y1) / 2))
    pts = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return cv.create_polygon(pts, smooth=True, **kw)


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------


class Card(tk.Frame):
    """A white rounded panel with a hairline border.

    Children go in ``self.body``, inset by at least ``RADIUS`` so their square
    corners stay inside the rounded silhouette and no seams show.
    """

    RADIUS = 12

    def __init__(self, master, pad_x: int = 20, pad_y: int = 13, **kw):
        super().__init__(master, bg=BG, **kw)
        self._bg = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0, takefocus=0)
        self._bg.place(x=0, y=0, relwidth=1, relheight=1)
        self.body = tk.Frame(self, bg=CARD)
        self.body.pack(fill="both", expand=True,
                       padx=max(pad_x, self.RADIUS), pady=max(pad_y, self.RADIUS))
        self.bind("<Configure>", self._redraw)

    def _redraw(self, event) -> None:
        self._bg.delete("all")
        round_rect(self._bg, 1, 1, event.width - 1, event.height - 1,
                   self.RADIUS, fill=CARD, outline=BORDER, width=1)


def divider(parent, pady=(10, 9)) -> tk.Frame:
    """A 1-pixel hairline rule that fills its parent."""
    line = tk.Frame(parent, bg=DIVIDER, height=1)
    line.pack(fill="x", pady=pady)
    return line


def section(parent, text: str, font) -> tk.Label:
    """Small upper-case card heading."""
    return tk.Label(parent, text=text.upper(), bg=CARD, fg=MUTED, font=font, anchor="w")


# ---------------------------------------------------------------------------
# Button
# ---------------------------------------------------------------------------


class Button(tk.Canvas):
    """Flat pill button with hover / press / disabled states.

    ``kind`` is one of ``primary`` (filled accent), ``secondary`` (outlined) or
    ``ghost`` (text only until hovered).
    """

    def __init__(self, master, text, command=None, kind="secondary", font=None,
                 height=36, pad_x=18, bg=CARD):
        width = font.measure(text) + 2 * pad_x
        super().__init__(master, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0, takefocus=0)
        self._font = font
        self._text, self._command, self._kind = text, command, kind
        self._cw, self._ch = width, height
        self._enabled, self._hover, self._down = True, False, False
        self.bind("<Enter>", lambda _e: self._set(hover=True))
        self.bind("<Leave>", lambda _e: self._set(hover=False, down=False))
        self.bind("<Button-1>", lambda _e: self._set(down=True))
        self.bind("<ButtonRelease-1>", self._release)
        self._draw()

    # -- state ---------------------------------------------------------------
    def _set(self, **kw) -> None:
        self._hover = kw.get("hover", self._hover)
        self._down = kw.get("down", self._down)
        self.configure(cursor="hand2" if (self._enabled and self._hover) else "")
        self._draw()

    def _release(self, event) -> None:
        fire = self._down and self._enabled and 0 <= event.x <= self._cw
        self._down = False
        self._draw()
        if fire and self._command:
            self._command()

    def set_enabled(self, value: bool) -> None:
        self._enabled = bool(value)
        self._down = False
        self._draw()

    # -- paint ---------------------------------------------------------------
    def _palette(self):
        if not self._enabled:
            return {
                "primary": (ACCENT_DIM, "", "#ffffff"),
                "secondary": (CARD, BORDER, FAINT),
                "ghost": (self["bg"], "", FAINT),
            }[self._kind]
        pressed, hovered = self._down, self._hover
        if self._kind == "primary":
            fill = ACCENT_D if pressed else (ACCENT_D if hovered else ACCENT)
            return (fill, "", "#ffffff")
        if self._kind == "secondary":
            fill = "#eef0f3" if pressed else (HOVER if hovered else CARD)
            return (fill, BORDER_D if hovered else BORDER, BODY)
        fill = "#e9ebef" if pressed else (HOVER if hovered else self["bg"])
        return (fill, "", MUTED if not hovered else BODY)

    def _draw(self) -> None:
        self.delete("all")
        fill, outline, fg = self._palette()
        kw = {"fill": fill}
        if outline:
            kw.update(outline=outline, width=1)
        else:
            kw.update(outline=fill)
        round_rect(self, 1, 1, self._cw - 1, self._ch - 1, min(9, self._ch // 2), **kw)
        self.create_text(self._cw / 2, self._ch / 2 + 0.5, text=self._text,
                         fill=fg, font=self._font)


# ---------------------------------------------------------------------------
# Segmented switch
# ---------------------------------------------------------------------------


class Segmented(tk.Canvas):
    """iOS-style segmented control: a track with a sliding white pill.

    ``items`` is a list of ``(value, label)``. Any value in ``disabled`` is drawn
    greyed and refuses clicks -- used for GPU mode on a machine with no
    accelerator.
    """

    def __init__(self, master, items, variable: tk.StringVar, command=None,
                 font=None, font_active=None, height=36, seg_width=None,
                 bg=CARD, pad_x=18):
        self._items = list(items)
        self._var = variable
        self._command = command
        self._font, self._font_active = font, font_active or font
        self._disabled: set[str] = set()
        widest = max(self._font_active.measure(lbl) for _v, lbl in self._items)
        self._seg = seg_width or (widest + 2 * pad_x)
        width = self._seg * len(self._items)
        super().__init__(master, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0, takefocus=0)
        self._cw, self._ch = width, height
        self._hover = -1
        # Follow the variable, not just clicks: the app also sets it directly
        # (forcing CPU when no GPU exists, restoring a choice), and a control
        # that silently disagrees with its own variable is worse than none.
        self._var.trace_add("write", self._on_var)
        self.bind("<Button-1>", self._click)
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda _e: self._set_hover(-1))
        self._draw()

    # -- state ---------------------------------------------------------------
    def _on_var(self, *_args) -> None:
        if self.winfo_exists():
            self._draw()

    def set_disabled(self, values) -> None:
        self._disabled = set(values)
        self._draw()

    def _index_at(self, x) -> int:
        i = int(x // self._seg)
        return i if 0 <= i < len(self._items) else -1

    def _set_hover(self, i) -> None:
        if i != self._hover:
            self._hover = i
            enabled = i >= 0 and self._items[i][0] not in self._disabled
            self.configure(cursor="hand2" if enabled else "")
            self._draw()

    def _motion(self, event) -> None:
        self._set_hover(self._index_at(event.x))

    def _click(self, event) -> None:
        i = self._index_at(event.x)
        if i < 0:
            return
        value = self._items[i][0]
        if value in self._disabled or value == self._var.get():
            return
        self._var.set(value)
        self._draw()
        if self._command:
            self._command()

    # -- paint ---------------------------------------------------------------
    def _draw(self) -> None:
        self.delete("all")
        round_rect(self, 1, 1, self._cw - 1, self._ch - 1, 10,
                   fill=SEG_TRACK, outline=BORDER, width=1)
        current = self._var.get()
        for i, (value, label) in enumerate(self._items):
            x0 = i * self._seg
            active = value == current
            off = value in self._disabled
            if active:
                round_rect(self, x0 + 3, 3, x0 + self._seg - 3, self._ch - 3, 8,
                           fill=CARD, outline=BORDER_D, width=1)
            elif i == self._hover and not off:
                round_rect(self, x0 + 3, 3, x0 + self._seg - 3, self._ch - 3, 8,
                           fill="#e7e9ed", outline="#e7e9ed")
            fg = FAINT if off else (INK if active else MUTED)
            self.create_text(x0 + self._seg / 2, self._ch / 2 + 0.5, text=label,
                             fill=fg, font=self._font_active if active else self._font)


# ---------------------------------------------------------------------------
# Check box
# ---------------------------------------------------------------------------


class Check(tk.Frame):
    """Square check box with a hand-drawn tick, plus a clickable label."""

    BOX = 18

    def __init__(self, master, text, variable: tk.BooleanVar, command=None,
                 font=None, bg=CARD):
        super().__init__(master, bg=bg)
        self._var, self._command = variable, command
        self._hover = False
        self._box = tk.Canvas(self, width=self.BOX, height=self.BOX, bg=bg,
                              highlightthickness=0, bd=0, takefocus=0)
        self._box.pack(side="left")
        self._label = tk.Label(self, text=text, bg=bg, fg=BODY, font=font,
                               anchor="w", justify="left")
        self._label.pack(side="left", padx=(9, 0))
        self._var.trace_add("write", self._on_var)   # see Segmented._on_var
        for widget in (self, self._box, self._label):
            widget.bind("<Button-1>", self._toggle)
            widget.bind("<Enter>", lambda _e: self._set_hover(True))
            widget.bind("<Leave>", lambda _e: self._set_hover(False))
            widget.configure(cursor="hand2")
        self._draw()

    def _on_var(self, *_args) -> None:
        if self.winfo_exists():
            self._draw()

    def _set_hover(self, value) -> None:
        self._hover = value
        self._draw()

    def _toggle(self, _event=None) -> None:
        self._var.set(not self._var.get())
        self._draw()
        if self._command:
            self._command()

    def _draw(self) -> None:
        cv, n = self._box, self.BOX
        cv.delete("all")
        on = bool(self._var.get())
        if on:
            round_rect(cv, 1, 1, n - 1, n - 1, 5, fill=ACCENT, outline=ACCENT_D)
            cv.create_line(4.5, 9.2, 7.6, 12.4, 13.4, 5.8, fill="#ffffff",
                           width=2, capstyle="round", joinstyle="round")
        else:
            round_rect(cv, 1, 1, n - 1, n - 1, 5, fill=FIELD,
                       outline=ACCENT if self._hover else BORDER_D)


# ---------------------------------------------------------------------------
# Numeric stepper
# ---------------------------------------------------------------------------


class Stepper(tk.Canvas):
    """``[ − ]  8  [ + ]`` -- a spinbox without the 3-D arrows."""

    def __init__(self, master, variable: tk.IntVar, lo=1, hi=999, font=None,
                 width=112, height=32, bg=CARD, command=None):
        super().__init__(master, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0, takefocus=0)
        self._var, self._lo, self._hi = variable, lo, hi
        self._font, self._cw, self._ch = font, width, height
        self._command, self._enabled = command, True
        self._zone = 32
        self._hover = 0
        self.bind("<Button-1>", self._click)
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda _e: self._set_hover(0))
        self.bind("<MouseWheel>", lambda e: self._bump(1 if e.delta > 0 else -1))
        self.bind("<Button-4>", lambda _e: self._bump(+1))
        self.bind("<Button-5>", lambda _e: self._bump(-1))
        self._draw()

    def set_enabled(self, value: bool) -> None:
        self._enabled = bool(value)
        self._draw()

    def set_bounds(self, lo: int, hi: int) -> None:
        """Narrow or widen the range without rebuilding the widget."""
        self._lo, self._hi = lo, hi
        self._draw()

    def set_variable(self, variable: tk.IntVar) -> None:
        """Point the same widget at a different variable, so one stepper can
        serve several settings without any of them losing its value."""
        if variable is not self._var:
            self._var = variable
            self._draw()

    def _value(self) -> int:
        try:
            return max(self._lo, min(self._hi, int(self._var.get())))
        except (tk.TclError, ValueError):
            return self._lo

    def _bump(self, delta: int) -> None:
        if not self._enabled:
            return
        self._var.set(max(self._lo, min(self._hi, self._value() + delta)))
        self._draw()
        if self._command:
            self._command()

    def _hit(self, x) -> int:
        if x < self._zone:
            return -1
        if x > self._cw - self._zone:
            return 1
        return 0

    def _motion(self, event) -> None:
        self._set_hover(self._hit(event.x) if self._enabled else 0)

    def _set_hover(self, value) -> None:
        if value != self._hover:
            self._hover = value
            self.configure(cursor="hand2" if value else "")
            self._draw()

    def _click(self, event) -> None:
        self._bump(self._hit(event.x))

    def _draw(self) -> None:
        self.delete("all")
        w, h, z = self._cw, self._ch, self._zone
        round_rect(self, 1, 1, w - 1, h - 1, 8,
                   fill=FIELD if self._enabled else CARD, outline=BORDER_D)
        for side, x0, x1 in ((-1, 1, z), (1, w - z, w - 1)):
            if self._enabled and self._hover == side:
                round_rect(self, x0, 1, x1, h - 1, 8, fill="#eceff3", outline="#eceff3")
        ink = BODY if self._enabled else FAINT
        self.create_line(z / 2 - 5, h / 2, z / 2 + 5, h / 2, fill=ink, width=2,
                         capstyle="round")
        self.create_line(w - z / 2 - 5, h / 2, w - z / 2 + 5, h / 2, fill=ink,
                         width=2, capstyle="round")
        self.create_line(w - z / 2, h / 2 - 5, w - z / 2, h / 2 + 5, fill=ink,
                         width=2, capstyle="round")
        self.create_line(z + 0.5, 6, z + 0.5, h - 6, fill=BORDER)
        self.create_line(w - z - 0.5, 6, w - z - 0.5, h - 6, fill=BORDER)
        self.create_text(w / 2, h / 2 + 0.5, text=str(self._value()),
                         fill=INK if self._enabled else FAINT, font=self._font)


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


class Bar(tk.Canvas):
    """Rounded progress track. ``set(pct)`` takes 0-100."""

    def __init__(self, master, height=10, bg=CARD):
        super().__init__(master, height=height, bg=bg, highlightthickness=0,
                         bd=0, takefocus=0)
        self._pct = 0.0
        self._ch = height
        self._colour = ACCENT
        self.bind("<Configure>", lambda _e: self._draw())

    def set(self, pct: float, colour: str | None = None) -> None:
        self._pct = max(0.0, min(100.0, float(pct)))
        if colour:
            self._colour = colour
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        w = self.winfo_width() or 1
        h = self._ch
        round_rect(self, 0, 0, w, h, h / 2, fill=TRACK, outline=TRACK)
        if self._pct > 0:
            end = max(h, w * self._pct / 100.0)
            round_rect(self, 0, 0, end, h, h / 2,
                       fill=self._colour, outline=self._colour)


# ---------------------------------------------------------------------------
# Small compound widgets
# ---------------------------------------------------------------------------


class Stat(tk.Frame):
    """A number beside its caption, used for the counters strip.

    Side by side rather than stacked: the strip sits above the log pane, and
    every row it does not take is a line of log the reader gets instead.
    """

    def __init__(self, master, caption, font_value, font_caption, bg=CARD):
        super().__init__(master, bg=bg)
        self._value = tk.Label(self, text="0", bg=bg, fg=INK, font=font_value,
                               anchor="w")
        self._value.pack(side="left")
        tk.Label(self, text=caption, bg=bg, fg=MUTED, font=font_caption,
                 anchor="w").pack(side="left", padx=(6, 0), pady=(4, 0))

    def set(self, text, colour: str = INK) -> None:
        self._value.configure(text=text, fg=colour)


class Pill(tk.Canvas):
    """Status chip: a coloured dot plus a short label."""

    def __init__(self, master, font, bg=BG, height=26):
        super().__init__(master, height=height, width=10, bg=bg,
                         highlightthickness=0, bd=0, takefocus=0)
        self._font, self._ch = font, height
        self._text, self._colour, self._tint = "", MUTED, "#ecedf0"

    def set(self, text: str, colour: str = MUTED, tint: str = "#ecedf0") -> None:
        self._text, self._colour, self._tint = text, colour, tint
        width = self._font.measure(text) + 42
        self.configure(width=width)
        self.delete("all")
        round_rect(self, 1, 1, width - 1, self._ch - 1, self._ch / 2,
                   fill=tint, outline=tint)
        cy = self._ch / 2
        self.create_oval(14, cy - 3.5, 21, cy + 3.5, fill=colour, outline=colour)
        self.create_text(28, cy + 0.5, text=text, fill=colour, font=self._font,
                         anchor="w")


# ---------------------------------------------------------------------------
# Text field and scrollbar
# ---------------------------------------------------------------------------


class Field(tk.Frame):
    """Bordered container around a borderless ``tk.Entry``.

    ``tk.Entry`` has no padding option, so the frame owns the border and the
    focus ring while the entry itself sits inset.
    """

    def __init__(self, master, textvariable, font, bg=CARD, pad=(11, 8)):
        super().__init__(master, bg=FIELD, highlightthickness=1,
                         highlightbackground=BORDER_D, highlightcolor=BORDER_D,
                         bd=0)
        self.entry = tk.Entry(self, textvariable=textvariable, font=font,
                              relief="flat", bg=FIELD, fg=INK, bd=0,
                              highlightthickness=0, insertbackground=ACCENT,
                              selectbackground=ACCENT_SOFT, selectforeground=INK)
        self.entry.pack(fill="both", expand=True, padx=pad[0], pady=pad[1])
        self.entry.bind("<FocusIn>", lambda _e: self.configure(
            highlightbackground=ACCENT, highlightcolor=ACCENT))
        self.entry.bind("<FocusOut>", lambda _e: self.configure(
            highlightbackground=BORDER_D, highlightcolor=BORDER_D))


class Scrollbar(tk.Canvas):
    """Minimal scrollbar: a rounded thumb on an invisible track, and nothing
    at all when the content fits -- which is what a log pane wants."""

    MIN_THUMB = 26

    def __init__(self, master, command, width=10, bg=FIELD):
        # height=1, not the Canvas default: this is packed ``fill="y"`` and
        # stretches to whatever it is given, so a real requested height would
        # only inflate the layout -- the default 265 px did exactly that, and
        # squeezed the log pane it belongs to down to three lines.
        super().__init__(master, width=width, height=1, bg=bg,
                         highlightthickness=0, bd=0, takefocus=0)
        self._command = command
        self._first, self._last = 0.0, 1.0
        self._cw = width
        self._drag_y = None
        self._drag_first = 0.0
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag_y", None))

    def set(self, first, last) -> None:
        """Called by the Text widget through ``yscrollcommand``."""
        self._first, self._last = float(first), float(last)
        self._draw()

    def _press(self, event) -> None:
        self._drag_y = event.y
        self._drag_first = self._first

    def _drag(self, event) -> None:
        if self._drag_y is None:
            return
        height = self.winfo_height() or 1
        delta = (event.y - self._drag_y) / height
        self._command("moveto", max(0.0, min(1.0, self._drag_first + delta)))

    def _draw(self) -> None:
        self.delete("all")
        if self._last - self._first >= 0.999:
            return  # everything fits: no thumb at all
        height = self.winfo_height() or 1
        top = self._first * height
        bottom = max(self._last * height, top + self.MIN_THUMB)
        if bottom > height:
            top, bottom = height - (bottom - top), height
        round_rect(self, 2, top + 1, self._cw - 2, bottom - 1,
                   (self._cw - 4) / 2, fill="#ccd2d9", outline="#ccd2d9")
