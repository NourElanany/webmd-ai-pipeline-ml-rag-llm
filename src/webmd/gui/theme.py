# ============================================================
# gui/theme.py — Single source of truth for all visual constants
# and ttk style configuration
# ============================================================

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from webmd.config import COLORS, FONTS

# ── Convenience aliases (used throughout gui/ modules) ───────────────────────
DARK_BG  = COLORS["dark_bg"]
PANEL_BG = COLORS["panel_bg"]
ACCENT   = COLORS["accent"]
ACCENT2  = COLORS["accent2"]
TEXT     = COLORS["text"]
SUBTEXT  = COLORS["subtext"]
CARD_BG  = COLORS["card_bg"]
GREEN    = COLORS["green"]
RED      = COLORS["red"]
YELLOW   = COLORS["yellow"]
BORDER   = COLORS["border"]
BLUE     = COLORS["blue"]
ORANGE   = COLORS["orange"]

FONT_TITLE  = FONTS["title"]
FONT_BOLD   = FONTS["bold"]
FONT_NORMAL = FONTS["normal"]
FONT_SMALL  = FONTS["small"]
FONT_MONO   = FONTS["mono"]


def apply_style(style: ttk.Style) -> None:
    """Apply the dark Catppuccin-inspired theme to all ttk widgets.

    Call once after creating the root Tk window.
    """
    style.theme_use("clam")

    style.configure("TNotebook",        background=DARK_BG,  borderwidth=0)
    style.configure("TNotebook.Tab",    background=PANEL_BG, foreground=TEXT,
                    padding=[16, 7],    font=FONT_BOLD)
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#fff")])

    style.configure("TFrame",           background=DARK_BG)

    style.configure("Treeview",         background=CARD_BG,  foreground=TEXT,
                    fieldbackground=CARD_BG, rowheight=28,   font=FONT_MONO)
    style.configure("Treeview.Heading", background=PANEL_BG, foreground=ACCENT,
                    font=FONT_BOLD)
    style.map("Treeview",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#fff")])

    style.configure("Vertical.TScrollbar",
                    background=PANEL_BG, troughcolor=DARK_BG,
                    arrowcolor=SUBTEXT,  borderwidth=0)
    style.configure("Horizontal.TScrollbar",
                    background=PANEL_BG, troughcolor=DARK_BG,
                    arrowcolor=SUBTEXT,  borderwidth=0)
