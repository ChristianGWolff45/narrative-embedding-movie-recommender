"""Shared visual language for every scene.

Import colors, fonts and sizes from here; never hardcode them in a scene.
"""
from manim import BLUE_C, GREY_B, ORANGE, TEAL_C, WHITE, config

BG = "#0f1117"
config.background_color = BG

FG = WHITE
MUTED = GREY_B
ACCENT = TEAL_C      # the query / user-taste vector
HIGHLIGHT = ORANGE   # the recommended movie
SECONDARY = BLUE_C   # other movies in embedding space

FONT = "DejaVu Sans"  # a system font every teammate has installed
TITLE_SIZE = 44
BODY_SIZE = 28
