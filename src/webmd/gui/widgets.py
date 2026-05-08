# ============================================================
# gui/widgets.py — Reusable Tkinter widget helpers
# All widget factories live here; tabs import from this module only.
# ============================================================

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

from webmd.gui.theme import (
    ACCENT, CARD_BG, DARK_BG, FONT_BOLD, FONT_NORMAL,
    FONT_SMALL, PANEL_BG, RED, SUBTEXT, TEXT,
    COLORS,
)


def scrolled_text(parent: tk.Widget, **kw) -> tuple[tk.Frame, tk.Text]:
    """Return (frame, text_widget) with a vertical scrollbar attached."""
    frame = tk.Frame(parent, bg=DARK_BG)
    txt = tk.Text(
        frame, bg=CARD_BG, fg=TEXT, font=FONT_NORMAL,
        relief="flat", wrap="word", padx=14, pady=10,
        insertbackground=TEXT, selectbackground=ACCENT,
        selectforeground="#fff", **kw,
    )
    sb = ttk.Scrollbar(frame, command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    txt.pack(fill="both", expand=True)
    return frame, txt


def card_label(
    parent: tk.Widget,
    title: str,
    value: str,
    color: str = ACCENT,
    col: int = 0,
    row: int = 0,
    colspan: int = 1,
) -> None:
    """Place a dark card with a small title and large coloured value into a grid."""
    f = tk.Frame(parent, bg=CARD_BG, padx=16, pady=12)
    f.grid(row=row, column=col, columnspan=colspan, padx=8, pady=6, sticky="nsew")
    tk.Label(f, text=title, bg=CARD_BG, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w")
    tk.Label(f, text=value, bg=CARD_BG, fg=color,
             font=("Segoe UI", 15, "bold")).pack(anchor="w")


def section_label(parent: tk.Widget, text: str) -> None:
    """Pack a section heading label (accent2 colour)."""
    tk.Label(
        parent, text=text, bg=DARK_BG, fg=COLORS["accent2"], font=FONT_BOLD,
    ).pack(anchor="w", padx=16, pady=(14, 4))


def btn(
    parent: tk.Widget,
    text: str,
    cmd,
    bg: str = ACCENT,
    fg: str = "#fff",
    **kw,
) -> tk.Button:
    """Return a flat, cursor-hand styled button."""
    return tk.Button(
        parent, text=text, command=cmd, bg=bg, fg=fg,
        font=FONT_BOLD, relief="flat", cursor="hand2",
        activebackground=COLORS["accent_active"], activeforeground="#fff",
        padx=16, pady=6, **kw,
    )


def plot_viewer(parent: tk.Widget, plots_dict: dict[str, Path]) -> None:
    """Sidebar chart list + right-side image viewer.

    *plots_dict* maps display name → Path.  Only plots whose file exists are
    shown in the sidebar.  The first available plot is displayed automatically
    after a short delay so the window has time to render.
    """
    left = tk.Frame(parent, bg=PANEL_BG, width=200)
    left.pack(side="left", fill="y")
    left.pack_propagate(False)
    tk.Label(left, text="Charts", bg=PANEL_BG, fg=ACCENT,
             font=FONT_BOLD).pack(pady=(14, 6), padx=10)

    right = tk.Frame(parent, bg=DARK_BG)
    right.pack(side="right", fill="both", expand=True)
    img_lbl = tk.Label(right, bg=DARK_BG)
    img_lbl.pack(fill="both", expand=True, padx=8, pady=8)

    _ref: list = [None]
    _cur: list = [None]

    def _show(path: Path) -> None:
        _cur[0] = path
        try:
            img = Image.open(path)
            w = max(right.winfo_width() - 16, 800)
            h = max(right.winfo_height() - 16, 500)
            img.thumbnail((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            _ref[0] = photo
            img_lbl.configure(image=photo, text="")
        except Exception as ex:
            img_lbl.configure(text=str(ex), fg=RED, image="")

    right.bind("<Configure>", lambda e: _cur[0] and _show(_cur[0]))

    bkw = dict(
        bg=CARD_BG, fg=TEXT, activebackground=ACCENT, activeforeground="#fff",
        relief="flat", font=FONT_SMALL, cursor="hand2", anchor="w", padx=10, pady=5,
    )
    first: list = [None]
    for name, path in plots_dict.items():
        if Path(path).exists():
            tk.Button(left, text=f"  {name}", **bkw,
                      command=lambda p=path: _show(p)).pack(fill="x", padx=6, pady=2)
            if first[0] is None:
                first[0] = path

    if first[0]:
        left.after(300, lambda: _show(first[0]))


def make_entry(
    parent: tk.Widget,
    var: tk.StringVar,
    root: tk.Widget,
    font: tuple = FONT_NORMAL,
    width: int | None = None,
    **grid_kw,
) -> tk.Entry:
    """Create a styled flat Entry with Ctrl+V paste and right-click context menu.

    *root* is needed for clipboard access in the paste handler.
    """
    kw: dict = dict(
        textvariable=var, bg=CARD_BG, fg=TEXT, insertbackground=TEXT,
        font=font, relief="flat", bd=0,
        highlightthickness=1,
        highlightbackground=COLORS["border"],
        highlightcolor=ACCENT,
    )
    if width is not None:
        kw["width"] = width
    e = tk.Entry(parent, **kw)
    e.grid(**grid_kw)
    bind_paste(e, root)
    _bind_context_menu(e, root)
    return e


def bind_paste(widget: tk.Entry, root: tk.Widget) -> None:
    """Bind Ctrl+V to a reliable paste handler that works on Windows dark entries."""
    def _paste(event: tk.Event) -> str:
        try:
            text = root.clipboard_get()
            try:
                widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                pass
            widget.insert(tk.INSERT, text)
        except tk.TclError:
            pass
        return "break"

    widget.bind("<Control-v>", _paste)
    widget.bind("<Control-V>", _paste)


def _bind_context_menu(widget: tk.Entry, root: tk.Widget) -> None:
    """Attach a right-click context menu (Paste / Copy / Cut / Select All)."""
    def _show(event: tk.Event) -> None:
        m = tk.Menu(
            root, tearoff=0, bg=CARD_BG, fg=TEXT,
            activebackground=ACCENT, activeforeground="#fff", relief="flat",
        )
        m.add_command(label="Paste",      command=lambda: widget.event_generate("<<Paste>>"))
        m.add_command(label="Copy",       command=lambda: widget.event_generate("<<Copy>>"))
        m.add_command(label="Cut",        command=lambda: widget.event_generate("<<Cut>>"))
        m.add_command(label="Select All", command=lambda: widget.select_range(0, "end"))
        try:
            m.tk_popup(event.x_root, event.y_root)
        finally:
            m.grab_release()

    widget.bind("<Button-3>", _show)
