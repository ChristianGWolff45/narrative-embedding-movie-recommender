# Manim Best Practices

Last updated 2026-11-19. Editable copy: [Claude Doc](https://claude.ai/artifact/Wu6WbD7iSWn3FyPHV4rkWb).

## Setup

Manim Community Edition 0.21.0 and manim-slides 5.7.0 are installed in their own uv project at `presentation/`, separate from the recommender. They must stay separate: Manim 0.19+ needs numpy ≥ 2.1, while the recommender pins numpy < 2.

| Package | Version | Why pinned |
| --- | --- | --- |
| manim | 0.21.0 (`>=0.19,<0.22`) | 0.22 removed a config option that manim-slides 5.7 still sets, so every render crashes |
| manim-slides[pyside6] | 5.7.0 | Live presenting with click-to-advance; PySide6 is the Qt backend for its player |
| numpy | 2.x | Pulled in by Manim; never install into the root `.venv` |

System dependencies were already present on this machine: Cairo and Pango dev headers, ffmpeg, and a full TeX Live (needed for `MathTex`/`Tex`). A teammate on a fresh Ubuntu box needs `sudo apt install libcairo2-dev libpango1.0-dev ffmpeg texlive-latex-extra texlive-fonts-extra dvisvgm`.

Everyday commands, run from `presentation/`:

```bash
uv sync                                          # first time on a new machine
uv run manim -pql scenes/smoke_test.py SmokeTest # draft render, 480p15, opens preview
uv run manim -qh scenes/smoke_test.py SmokeTest  # final render, 1080p60
uv run manim-slides SmokeTest                    # present it (→ next, ← previous)
```

Quality flags: `-ql` 480p15 for drafting, `-qm` 720p30, `-qh` 1080p60 for the real talk, `-qk` 4K only if the projector supports it. Add `-p` to open the result when done.

## Project structure

Keep one scene file per talk section, with all shared styling in `scenes/style.py`. Rendered output stays out of git.

```text
presentation/
├── pyproject.toml      # manim + manim-slides, isolated from the recommender
├── uv.lock             # commit this so everyone renders with identical versions
├── .gitignore          # .venv/, media/, slides/, __pycache__/
├── assets/             # SVGs, posters, exported plots (add when needed)
└── scenes/
    ├── style.py        # palette, font, sizes, background
    ├── smoke_test.py   # install check; delete once real scenes exist
    ├── s01_problem.py
    ├── s02_embeddings.py
    └── s03_results.py
```

- `style.py` lives inside `scenes/` on purpose: Manim puts the scene file's own folder on the import path, so `from style import ACCENT` works with no `PYTHONPATH` tweaks.
- Prefix files with `s01_`, `s02_` so the talk order matches the alphabetical order of the files.
- One `Slide` class per file is easiest to split between teammates and avoids merge conflicts. Use several only for short, tightly linked beats.
- Pull real numbers (evaluation metrics, example movies) from the recommender's output files, not hand-typed values, so the slides cannot drift from the results.
- Never commit `media/` or `slides/`: they are large and can always be rebuilt.

## Presentation workflow

Present live with manim-slides, not by playing MP4s: each click advances to the next animation, so the speaker controls the pace. Subclass `Slide` instead of `Scene` and call `self.next_slide()` wherever you would click.

```python
from manim_slides import Slide

class Embeddings(Slide):
    def construct(self):
        self.play(Write(title))
        self.next_slide()                 # pause: wait for click
        self.next_slide(loop=True)        # this block loops until click
        self.play(Rotate(cloud, TAU), run_time=6, rate_func=linear)
        self.next_slide()
        self.play(FadeOut(cloud))
```

1. Render each section at final quality: `uv run manim -qh scenes/s01_problem.py Problem`.
2. Rehearse in the player: `uv run manim-slides Problem Embeddings Results` plays them in order. Keys: → next, ← previous, Space pause, R replay, F full screen, H hide mouse, Q quit.
3. Make an offline backup: `uv run manim-slides convert Problem Embeddings Results talk.html --one-file --offline` writes one reveal.js page that needs no internet. The `pdf` and `pptx` formats also exist; PowerPoint export may need the `manim-slides[pptx]` extra.
4. Bring both the HTML and the `slides/` folder on a USB stick, and test on the venue projector's resolution beforehand.

Guidelines:

- Put a `next_slide()` wherever you would speak for more than a sentence. Don't animate through ideas faster than you can say them.
- End each section on a calm still frame, so the pause before the next click looks deliberate.
- Use `loop=True` only for ambient motion (a rotating 3D embedding cloud). Never loop something that carries information.
- Start each section with `self.wait(0.1)` or a short fade-in so the first frame isn't blank while the projector syncs.

## Animation & design principles

Every animation should show a change the audience needs to see; if it is only decoration, cut it. A consistent visual language does more for clarity than fancy effects.

**One meaning per color.** The palette in `scenes/style.py` assigns roles. Keep them fixed across the whole talk:

| Constant | Color | Means |
| --- | --- | --- |
| `ACCENT` | teal | the user / query taste vector |
| `HIGHLIGHT` | orange | the recommended movie, the answer |
| `SECONDARY` | blue | the rest of the catalog |
| `MUTED` | grey | context, axes, de-emphasized items |
| `BG` | #0f1117 | background (dark reads best on projectors) |

**Layout by relation, not coordinates.** Position objects relative to each other so a change ripples correctly:

```python
title.to_edge(UP)
formula.next_to(title, DOWN, buff=0.4)
row = VGroup(card_a, card_b, card_c).arrange(RIGHT, buff=0.5)
row.next_to(formula, DOWN, buff=0.6)
legend.to_corner(DR)
```

The frame is 14.2 × 8 units. Leave about 0.5 units of margin; projectors crop edges.

**Pacing.**

- Default `run_time` is 1 s. Use 0.5–1 s for small moves and 1.5–2.5 s for the key reveal, but rarely anything longer.
- Use `LaggedStart(..., lag_ratio=0.05–0.1)` to introduce many objects. Showing 500 dots at once reads as noise.
- Move one thing at a time when it matters. Dim the rest with `.animate.set_opacity(0.25)` instead of removing it, so context stays.
- Prefer `Transform`/`ReplacementTransform` to show that two things are the same idea in different forms. Use `FadeOut` + `FadeIn` only when they are not.

**Group and name things.** Build compound objects as a `VGroup` subclass or a small factory function (`def movie_card(title, year) -> VGroup`), so scenes read like a script, not geometry.

**Clean up.** Fade out or reuse everything before the next section. Leftover objects stack up invisibly and slow later renders.

## Text & math

Use `Text` for words and `MathTex` for formulas; reach for `Tex` only when a sentence mixes both.

| Class | Renders with | Use for |
| --- | --- | --- |
| `Text` | Pango (system fonts) | titles, labels, movie names, anything non-math |
| `MarkupText` | Pango markup | one word bold or colored inside a label |
| `MathTex` | LaTeX, math mode | formulas: cosine similarity, loss functions, metrics |
| `Tex` | LaTeX, text mode | rare: a sentence with inline math |
| `Code` | Pygments | showing a code snippet on a slide |

- Always pass `font=FONT` and a size from `style.py`. Two font sizes per slide (title + body) is usually enough.
- Split a formula into substrings so parts can be colored and animated separately: `MathTex(r"\cos\theta", "=", r"\frac{u \cdot v}{\|u\|\|v\|}")`, then `formula[2].set_color(ACCENT)`.
- Color variables everywhere with `tex_to_color_map={"u": ACCENT, "v": HIGHLIGHT}`, matching the plot colors. It splits the string at every match, so a lone letter also hits inside commands like `\sum`; isolate variables with double braces, `{{u}}`, instead.
- Animate derivations with `TransformMatchingTex(old, new)`: unchanged pieces stay put and only the changed parts move. It needs the shared pieces to be separate substrings.
- Write raw strings (`r"..."`) for all LaTeX, or backslashes get eaten.
- Keep text short: a slide is not a document. Say the sentence; show the keyword.

LaTeX renders are cached in `media/Tex/`, so a formula costs time only the first time it appears.

## Data & ML visuals for the recommender

The core story (movies as points, a user as a vector, recommendations as nearest neighbours) maps directly onto Manim primitives. Show real data from the pipeline, reduced to 2D or 3D.

**Embedding space.** Project the narrative embeddings to 2D (PCA or UMAP) offline, save to a small `.npy`/CSV, and load it in the scene. Never run the embedding model inside a render.

```python
coords = np.load(ASSETS / "embeddings_2d.npy")          # (n, 2), precomputed
axes = Axes(x_range=[-1, 1], y_range=[-1, 1], x_length=6, y_length=6, tips=False)
dots = VGroup(*[Dot(axes.c2p(x, y), radius=0.04, color=SECONDARY) for x, y in coords])
```

Keep scatter plots to a few hundred dots; thousands of mobjects make renders slow and the slide unreadable. Sample, and keep the named movies you will talk about.

**Similarity as an angle.** Draw the user vector in `ACCENT` and candidate movies as arrows; show cosine similarity with `Angle(line_a, line_b)` and a live number:

```python
theta = ValueTracker(PI / 3)
movie = always_redraw(lambda: Arrow(axes.c2p(0, 0),
    axes.c2p(np.cos(theta.get_value()), np.sin(theta.get_value())), buff=0, color=HIGHLIGHT))
score = always_redraw(lambda: DecimalNumber(np.cos(theta.get_value() - user_angle))
    .next_to(movie.get_end(), UR))
self.play(theta.animate.set_value(user_angle + 0.1), run_time=2)
```

`ValueTracker` + `always_redraw` (or `.add_updater`) is the standard way to tie a number to a moving object. Remove updaters with `clear_updaters()` when done.

**Nearest neighbours.** Grow a `Circle` around the query, highlight the dots inside with `HIGHLIGHT`, and dim the rest. Then `Transform` those dots into a ranked list of movie cards.

**Results.** Use `BarChart` or hand-built `Rectangle` bars for evaluation metrics (single vs multi-embedding), with values loaded from the evaluation output. Grow bars with `GrowFromEdge(bar, DOWN)`, one model at a time.

**3D, sparingly.** `ThreeDScene` + `ThreeDAxes` + `self.begin_ambient_camera_rotation(rate=0.1)` makes a strong "embedding cloud" moment. Use it once; 3D labels are hard to read and renders are slower. For manim-slides, combine with `ThreeDSlide`.

## Performance & iteration

Draft at `-ql`, check single frames, and render `-qh` only once a section is locked. A 480p draft renders in seconds; 1080p60 has 4× the frames and 5× the pixels, so it takes many times longer.

| Goal | Command |
| --- | --- |
| Quick look at a draft | `uv run manim -pql scenes/s02_embeddings.py Embeddings` |
| Check layout without animating | `uv run manim -s -ql …` (saves the last frame as PNG; see pitfall below for `Slide` classes) |
| Jump to the part you're editing | `-n 12,15` renders only animations 12 to 15 |
| Force a clean render | `--disable_caching` (after changing assets or `style.py`) |
| Final | `uv run manim -qh …` |

- Manim caches each `play()` call as a partial movie keyed by its content, so re-rendering after a small edit only re-renders the animations that changed. Keep `media/` between runs.
- Temporarily put `self.next_section(skip_animations=True)` before finished parts of a long scene to skip them while you work on the end.
- Updaters and `always_redraw` run every frame. Ten of them on hundreds of dots will crawl; precompute positions or animate a `VGroup` instead.
- Stay on the default Cairo renderer. `--renderer=opengl` previews interactively but has gaps (some mobjects, manim-slides output) and is not worth the debugging for a talk.
- Render final sections in parallel from separate terminals; each file renders independently.

## Common pitfalls

The first three bit us while setting up; the rest are the usual Manim traps.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `AttributeError: ... no attribute 'write_to_movie'` on every render | manim 0.22 with manim-slides 5.7 | Keep the `manim<0.22` pin until manim-slides releases support |
| uv says manim and the recommender are "incompatible" | numpy ≥ 2.1 vs numpy < 2 | Run everything from `presentation/`, never add manim to the root project |
| `-s` saves the PNG, then fails with "Cannot merge an empty list of files" | manim-slides expects video output | Harmless; the PNG in `media/images/` is fine. Or test layout on a plain `Scene` |
| `ModuleNotFoundError: style` | Helper module outside `scenes/` | Keep shared modules next to the scene files |
| LaTeX error with a cryptic log | Unescaped `\`, missing `r""`, or `&`/`%` in text | Use raw strings; check the `.log` path printed in the error |
| Object appears but never animates, or vanishes | Called `self.add` after `play`, or animated a copy | Animate the object itself; use `self.add` only for static setup |
| `Transform` leaves a ghost or the wrong object moves later | `Transform(a, b)` keeps `a` on screen as b's shape | Use `ReplacementTransform` when you'll keep referring to `b` |
| Text looks different on a teammate's machine | Missing font falls back silently | Use a font everyone has (`DejaVu Sans`) or ship the font in `assets/` |
| Slide jumps when advancing | Object moved between `next_slide()` calls without `play` | Make every visible change an animation |

## Pre-talk checklist

- [ ] Every scene imports colors, font and sizes from `style.py`; no hex codes in scene files
- [ ] Every number on screen comes from the pipeline's output files
- [ ] All sections rendered with `-qh` and watched end to end in `manim-slides`
- [ ] Text is readable from the back of the room: body font_size ≥ 24, ≤ ~12 words per slide
- [ ] Offline HTML export (`--one-file --offline`) tested on another laptop
- [ ] Projector resolution and aspect ratio (16:9) confirmed with the venue
