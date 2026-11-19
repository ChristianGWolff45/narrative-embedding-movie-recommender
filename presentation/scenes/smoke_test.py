"""Smoke test: checks LaTeX, fonts, axes and manim-slides all work.

Render:   uv run manim -ql scenes/smoke_test.py SmokeTest
Present:  uv run manim-slides SmokeTest
"""
import numpy as np
from manim import (
    DOWN, RIGHT, UP, Arrow, Axes, Create, Dot, FadeIn, GrowArrow,
    GrowFromCenter, LaggedStart, MathTex, Text, VGroup, Write,
)
from manim_slides import Slide

from style import ACCENT, BODY_SIZE, FONT, HIGHLIGHT, SECONDARY, TITLE_SIZE


class SmokeTest(Slide):
    def construct(self):
        title = Text("Cosine similarity", font=FONT, font_size=TITLE_SIZE).to_edge(UP)
        formula = MathTex(
            r"\cos\theta", "=", r"\frac{u \cdot v}{\|u\|\,\|v\|}"
        ).next_to(title, DOWN)
        self.play(Write(title), FadeIn(formula, shift=UP * 0.3))
        self.next_slide()

        axes = Axes(
            x_range=[-1, 1, 0.5], y_range=[-1, 1, 0.5], x_length=4, y_length=4
        ).to_edge(DOWN)
        rng = np.random.default_rng(0)
        dots = VGroup(*[
            Dot(axes.c2p(*rng.uniform(-0.9, 0.9, 2)), color=SECONDARY, radius=0.05)
            for _ in range(30)
        ])
        query = Arrow(axes.c2p(0, 0), axes.c2p(0.7, 0.5), buff=0, color=ACCENT)
        label = Text("you", font=FONT, font_size=BODY_SIZE, color=ACCENT)
        label.next_to(query.get_end(), RIGHT)

        self.play(Create(axes), LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.03))
        self.play(GrowArrow(query), FadeIn(label))
        self.next_slide()
        self.play(dots[3].animate.set_color(HIGHLIGHT).scale(2))
