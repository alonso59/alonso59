"""
CNN Step-by-Step Visualizer  —  Manim Community v0.20+

Shows each operation animated step by step:
  Input      1×32×32   — pixel grid excerpt
  Conv1      6×30×30   — sliding 3×3 kernel, dot-product per step
  ReLU                 — negative values clamped to 0
  MaxPool1   6×15×15   — 2×2 sliding window, max selection
  Conv2      16×12×12  — abbreviated sliding (multi-channel sum)
  MaxPool2   16×6×6    — abbreviated
  Flatten    576        — feature maps unrolling to vector
  FC1        120        — weighted connections, y = Wx + b
  FC2 / Out  84 → 10   — final projection
  Pipeline             — full diagram with shapes
"""

from manim import *
import numpy as np

# ── palette ──────────────────────────────────────────────────────────────────
BG      = BLACK
C_IN    = "#4FC3F7"   # light blue   – input
C_CONV  = "#FFD54F"   # amber        – kernels / conv output
C_RELU  = "#FF8A65"   # orange       – relu
C_POOL  = "#EF9A9A"   # rose         – pool
C_FC    = "#A5D6A7"   # green        – fully-connected
C_OUT   = "#CE93D8"   # purple       – output
C_GREY  = "#B0BEC5"   # dim grey
C_ARR   = "#90A4AE"   # arrow grey

rng = np.random.default_rng(42)


# ── grid helpers ──────────────────────────────────────────────────────────────

def make_grid(rows, cols, cs=0.44,
              fill=C_IN, fill_op=0.15,
              stroke=WHITE, sw=0.8):
    """Return (VGroup, 2-D list[Square]) centred at origin."""
    cells, grp = [], VGroup()
    for r in range(rows):
        row = []
        for c in range(cols):
            sq = Square(side_length=cs,
                        fill_color=fill, fill_opacity=fill_op,
                        stroke_color=stroke, stroke_width=sw)
            sq.move_to(RIGHT * c * cs + DOWN * r * cs)
            grp.add(sq)
            row.append(sq)
        cells.append(row)
    grp.center()
    return grp, cells


def val_labels(cells, vals, color=WHITE, fs=15):
    """VGroup of MathTex labels inside each cell."""
    g = VGroup()
    for r, row in enumerate(cells):
        for c, sq in enumerate(row):
            t = MathTex(str(int(vals[r][c])), color=color, font_size=fs)
            t.move_to(sq)
            g.add(t)
    return g


def patch_rect(cells, r0, c0, kr, kc, color=C_CONV, sw=2.5, fill_op=0.18):
    """Highlight rectangle over cells[r0:r0+kr, c0:c0+kc]."""
    tl = cells[r0][c0].get_corner(UL)
    br = cells[r0+kr-1][c0+kc-1].get_corner(DR)
    w, h = br[0]-tl[0], tl[1]-br[1]
    rect = Rectangle(width=w, height=h,
                     stroke_color=color, stroke_width=sw,
                     fill_color=color, fill_opacity=fill_op)
    rect.move_to((tl + br) / 2)
    return rect


def section_title(text, color, math=False):
    if math:
        return MathTex(text, color=color, font_size=28).to_edge(UP, buff=0.35)
    return Text(text, color=color, font_size=28).to_edge(UP, buff=0.35)


# ── main scene ────────────────────────────────────────────────────────────────

class CNNArchitecture(Scene):

    def construct(self):
        self.camera.background_color = BG
        self._title()
        self._input_scene()
        self._conv1_scene()
        self._relu_scene()
        self._pool1_scene()
        self._conv2_scene()
        self._pool2_scene()
        self._flatten_scene()
        self._fc_scene()
        self._pipeline_scene()

    # ── title ─────────────────────────────────────────────────────────────────
    def _title(self):
        t   = Text("CNN · Step by Step", color=WHITE, font_size=44)
        sub = MathTex(r"X \in \mathbb{R}^{1\times32\times32}",
                      color=C_IN, font_size=30).next_to(t, DOWN, buff=0.3)
        ops = MathTex(
            r"\text{Conv} \to \text{ReLU} \to \text{Pool}"
            r"\to \text{Conv} \to \text{Pool} \to \text{Flatten} \to \text{FC}",
            color=C_GREY, font_size=22).next_to(sub, DOWN, buff=0.25)
        self.play(Write(t))
        self.play(FadeIn(sub, shift=UP*0.2), FadeIn(ops, shift=UP*0.2))
        self.wait(1.5)
        self.play(FadeOut(VGroup(t, sub, ops)))

    # ── input ─────────────────────────────────────────────────────────────────
    def _input_scene(self):
        title = section_title("Input   X  ∈  ℝ¹×³²×³²", C_IN)
        vals  = rng.integers(0, 10, (6, 6))
        g, cells = make_grid(6, 6, 0.46, C_IN)
        g.shift(LEFT * 0.3)
        lbls = val_labels(cells, vals, C_IN, 15)
        shp  = MathTex(r"1\times32\times32", color=C_GREY, font_size=28)\
                       .next_to(g, RIGHT, buff=0.5)
        note = MathTex(r"(\text{6}\times\text{6 excerpt shown})",
                       color=C_GREY, font_size=20).next_to(shp, DOWN, buff=0.2)
        self.play(Write(title), run_time=0.5)
        self.play(Create(g), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(l) for l in lbls], lag_ratio=0.03),
                  run_time=0.8)
        self.play(Write(shp), Write(note))
        self.wait(1.2)
        self.play(FadeOut(VGroup(title, g, lbls, shp, note)))

    # ── conv1 — sliding 3×3 kernel ────────────────────────────────────────────
    def _conv1_scene(self):
        title = section_title(
            r"\text{Conv1}:\ K^{(1)}\in\mathbb{R}^{6\times1\times3\times3}"
            r"\;\Rightarrow\;6\times30\times30", C_CONV, math=True)

        G, K = 5, 3
        OUT  = G - K + 1   # 3×3 output excerpt

        x_vals = np.array([[3, 1, 0, 2, 1],
                           [0, 2, 1, 3, 0],
                           [1, 0, 3, 1, 2],
                           [2, 1, 0, 2, 1],
                           [0, 2, 1, 0, 3]])
        k_vals = np.array([[ 1,  0, -1],
                           [ 1,  0, -1],
                           [ 1,  0, -1]])

        CS = 0.45
        # input (left)
        ig, ic = make_grid(G, G, CS, C_IN)
        ig.shift(LEFT * 3.3)
        il = val_labels(ic, x_vals, C_IN, 14)
        il_lbl = MathTex(r"X[0,\cdot,\cdot]", color=C_IN, font_size=22)\
                         .next_to(ig, UP, buff=0.18)

        # kernel (centre)
        kg, kc = make_grid(K, K, CS, C_CONV, fill_op=0.25, stroke=C_CONV)
        kg.move_to(ORIGIN)
        kl = val_labels(kc, k_vals, C_CONV, 14)
        kl_lbl = MathTex(r"K^{(1)}[0,0,\cdot,\cdot]", color=C_CONV, font_size=22)\
                         .next_to(kg, UP, buff=0.18)

        # output (right)
        y_vals = np.array([[int(np.sum(x_vals[i:i+K, j:j+K]*k_vals))
                            for j in range(OUT)] for i in range(OUT)])
        og, oc = make_grid(OUT, OUT, CS, C_FC, fill_op=0.08, stroke=C_FC)
        og.shift(RIGHT * 3.3)
        ol_lbl = MathTex(r"Y_1[0,\cdot,\cdot]", color=C_FC, font_size=22)\
                         .next_to(og, UP, buff=0.18)

        formula = MathTex(
            r"Y[i,j]=\sum_{m,n}X[i{+}m,j{+}n]\cdot K[m,n]",
            color=WHITE, font_size=22).to_edge(DOWN, buff=0.35)

        self.play(Write(title), run_time=0.5)
        self.play(Create(ig), Create(kg), Create(og), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(l) for l in il], lag_ratio=0.02),
                  LaggedStart(*[FadeIn(l) for l in kl], lag_ratio=0.02),
                  run_time=0.5)
        self.play(Write(il_lbl), Write(kl_lbl), Write(ol_lbl))
        self.play(Write(formula))
        self.wait(0.3)

        # sliding animation
        patch    = None
        out_lbls = VGroup()
        for idx, (i, j) in enumerate([(i, j)
                                       for i in range(OUT)
                                       for j in range(OUT)]):
            new_p = patch_rect(ic, i, j, K, K, C_CONV)
            slow  = idx < 3

            if patch is None:
                patch = new_p
                self.play(FadeIn(patch), run_time=0.3)
            else:
                self.play(Transform(patch, new_p),
                          run_time=0.3 if slow else 0.12)

            val = y_vals[i, j]
            oc[i][j].set_fill(C_FC, opacity=0.5)
            lbl = MathTex(str(val), color=C_FC, font_size=14).move_to(oc[i][j])
            out_lbls.add(lbl)

            if slow:
                det = MathTex(rf"Y[{i},{j}]={val}",
                              color=C_GREY, font_size=18)\
                              .next_to(formula, UP, buff=0.15)
                self.play(FadeIn(lbl), FadeIn(det), run_time=0.2)
                self.wait(0.35)
                self.play(FadeOut(det), run_time=0.12)
            else:
                self.play(Flash(oc[i][j], color=C_FC,
                                flash_radius=CS*0.6, line_length=CS*0.28),
                          FadeIn(lbl), run_time=0.1)

        shp_note = MathTex(r"\text{shape: }6\times30\times30",
                           color=C_GREY, font_size=22)\
                           .next_to(og, DOWN, buff=0.25)
        self.play(Write(shp_note))
        self.wait(1.2)
        self.play(FadeOut(VGroup(
            title, ig, il, il_lbl,
            kg, kl, kl_lbl,
            og, out_lbls, ol_lbl,
            formula, shp_note, patch)))

    # ── relu ──────────────────────────────────────────────────────────────────
    def _relu_scene(self):
        title = section_title(r"\text{ReLU}:\ f(x)=\max(0,x)",
                               C_RELU, math=True)

        vals = np.array([[ 3, -1,  2, -2],
                         [-1,  4, -3,  1],
                         [ 2, -2,  5, -1],
                         [-3,  1, -1,  3]])
        g, cells = make_grid(4, 4, 0.52, C_RELU, fill_op=0.12, stroke=C_RELU)
        g.move_to(ORIGIN)
        lbls    = val_labels(cells, vals, WHITE, 18)
        formula = MathTex(r"f(x)=\max(0,x)", color=C_RELU, font_size=32)\
                          .to_edge(DOWN, buff=0.5)

        self.play(Write(title), run_time=0.4)
        self.play(Create(g), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(l) for l in lbls], lag_ratio=0.04),
                  run_time=0.5)
        self.play(Write(formula))
        self.wait(0.4)

        # clamp negatives to 0
        anims = []
        for r in range(4):
            for c in range(4):
                if vals[r][c] < 0:
                    cells[r][c].set_fill(C_RELU, opacity=0.02)
                    new_lbl = MathTex("0", color=C_GREY, font_size=18)\
                                      .move_to(cells[r][c])
                    anims.append(Transform(lbls[r*4+c], new_lbl))

        self.play(LaggedStart(*anims, lag_ratio=0.1), run_time=1.2)
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, g, lbls, formula)))

    # ── maxpool 1 ─────────────────────────────────────────────────────────────
    def _pool1_scene(self):
        title = section_title(
            r"\text{MaxPool }2\times2\;\Rightarrow\;6\times15\times15",
            C_POOL, math=True)

        vals = np.array([[3, 1, 2, 0],
                         [0, 4, 1, 3],
                         [2, 0, 5, 1],
                         [1, 3, 0, 4]])
        CS = 0.50

        ig, ic = make_grid(4, 4, CS, C_POOL, fill_op=0.12, stroke=C_POOL)
        ig.shift(LEFT * 2.8)
        il     = val_labels(ic, vals, C_POOL, 16)
        il_lbl = MathTex(r"Y_1[\cdot,\cdot,\cdot]", color=C_POOL,
                         font_size=22).next_to(ig, UP, buff=0.18)

        og, oc = make_grid(2, 2, CS*1.1, C_POOL, fill_op=0.08, stroke=C_POOL)
        og.shift(RIGHT * 2.8)
        ol_lbl = MathTex(r"\text{After Pool}", color=C_POOL,
                         font_size=22).next_to(og, UP, buff=0.18)

        formula = MathTex(
            r"Y[i,j]=\max_{m,n\in\{0,1\}}\,X[2i{+}m,\,2j{+}n]",
            color=WHITE, font_size=22).to_edge(DOWN, buff=0.35)

        self.play(Write(title), run_time=0.5)
        self.play(Create(ig), Create(og), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(l) for l in il], lag_ratio=0.04),
                  run_time=0.5)
        self.play(Write(il_lbl), Write(ol_lbl), Write(formula))
        self.wait(0.3)

        out_lbls = VGroup()
        for pi in range(2):
            for pj in range(2):
                p    = patch_rect(ic, pi*2, pj*2, 2, 2, C_POOL)
                win  = vals[pi*2:pi*2+2, pj*2:pj*2+2]
                mx   = int(win.max())
                mr, mc_ = divmod(int(win.argmax()), 2)
                mc_sq = ic[pi*2+mr][pj*2+mc_]

                self.play(FadeIn(p), run_time=0.3)
                self.play(mc_sq.animate.set_fill(C_POOL, opacity=0.7),
                          run_time=0.3)

                mx_lbl = MathTex(str(mx), color=C_POOL, font_size=18)\
                                 .move_to(mc_sq)
                self.play(FadeIn(mx_lbl), run_time=0.15)
                self.play(mx_lbl.animate.move_to(oc[pi][pj]),
                          oc[pi][pj].animate.set_fill(C_POOL, opacity=0.55),
                          run_time=0.4)
                out_lbls.add(mx_lbl)
                self.play(FadeOut(p), run_time=0.2)

        shp = MathTex(r"\text{shape: }6\times15\times15",
                      color=C_GREY, font_size=22).next_to(og, DOWN, buff=0.25)
        self.play(Write(shp))
        self.wait(1.2)
        self.play(FadeOut(VGroup(
            title, ig, il, il_lbl,
            og, out_lbls, ol_lbl,
            formula, shp)))

    # ── conv2 (abbreviated) ───────────────────────────────────────────────────
    def _conv2_scene(self):
        title = section_title(
            r"\text{Conv2}:\ K^{(2)}\in\mathbb{R}^{16\times6\times3\times3}"
            r"\;\Rightarrow\;16\times12\times12", C_CONV, math=True)

        G, K  = 5, 3
        OUT   = G - K + 1
        x_v   = rng.integers(0, 6, (G, G))
        k_v   = rng.integers(-1, 2, (K, K))
        CS    = 0.42

        ig, ic = make_grid(G, G, CS, C_IN)
        ig.shift(LEFT * 3.2)
        il     = val_labels(ic, x_v, C_IN, 13)
        il_lbl = MathTex(r"Y_1[c',\cdot,\cdot]", color=C_IN, font_size=20)\
                         .next_to(ig, UP, buff=0.15)

        kg, kc = make_grid(K, K, CS, C_CONV, fill_op=0.25, stroke=C_CONV)
        kg.move_to(ORIGIN)
        kl     = val_labels(kc, k_v, C_CONV, 13)
        kl_lbl = MathTex(r"K^{(2)}[c,c',\cdot,\cdot]", color=C_CONV, font_size=20)\
                         .next_to(kg, UP, buff=0.15)

        og, oc = make_grid(OUT, OUT, CS, C_FC, fill_op=0.08, stroke=C_FC)
        og.shift(RIGHT * 3.2)
        ol_lbl = MathTex(r"Y_2[c,\cdot,\cdot]", color=C_FC, font_size=20)\
                         .next_to(og, UP, buff=0.15)

        formula = MathTex(
            r"Y_2[c,i,j]=\sum_{c'}\sum_{m,n}Y_1[c',i{+}m,j{+}n]\cdot K^{(2)}[c,c',m,n]",
            color=WHITE, font_size=18).to_edge(DOWN, buff=0.35)

        self.play(Write(title), run_time=0.4)
        self.play(Create(ig), Create(kg), Create(og), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(l) for l in VGroup(*il, *kl)],
                               lag_ratio=0.02), run_time=0.5)
        self.play(Write(il_lbl), Write(kl_lbl), Write(ol_lbl), Write(formula))
        self.wait(0.3)

        patch    = None
        out_lbls = VGroup()
        for idx, (i, j) in enumerate([(i, j) for i in range(OUT)
                                               for j in range(OUT)]):
            new_p = patch_rect(ic, i, j, K, K, C_CONV)
            val   = int(np.sum(x_v[i:i+K, j:j+K] * k_v))
            if patch is None:
                patch = new_p
                self.play(FadeIn(patch), run_time=0.2)
            else:
                self.play(Transform(patch, new_p),
                          run_time=0.18 if idx < 2 else 0.10)
            oc[i][j].set_fill(C_FC, opacity=0.45)
            lbl = MathTex(str(val), color=C_FC, font_size=13).move_to(oc[i][j])
            out_lbls.add(lbl)
            self.play(FadeIn(lbl), run_time=0.08)

        shp = MathTex(r"\text{shape: }16\times12\times12",
                      color=C_GREY, font_size=22).next_to(og, DOWN, buff=0.25)
        self.play(Write(shp))
        self.wait(1.0)
        self.play(FadeOut(VGroup(
            title, ig, il, il_lbl,
            kg, kl, kl_lbl,
            og, out_lbls, ol_lbl,
            formula, shp, patch)))

    # ── maxpool 2 (abbreviated) ───────────────────────────────────────────────
    def _pool2_scene(self):
        title = section_title(
            r"\text{MaxPool }2\times2\;\Rightarrow\;16\times6\times6",
            C_POOL, math=True)

        vals = np.array([[5, 2, 3, 1],
                         [1, 3, 4, 2],
                         [0, 4, 1, 5],
                         [3, 1, 2, 3]])
        CS = 0.50
        ig, ic = make_grid(4, 4, CS, C_POOL, fill_op=0.12, stroke=C_POOL)
        ig.shift(LEFT * 2.8)
        il     = val_labels(ic, vals, C_POOL, 16)
        og, oc = make_grid(2, 2, CS*1.1, C_POOL, fill_op=0.08, stroke=C_POOL)
        og.shift(RIGHT * 2.8)
        formula = MathTex(
            r"Y[i,j]=\max_{m,n\in\{0,1\}}\,X[2i{+}m,\,2j{+}n]",
            color=WHITE, font_size=22).to_edge(DOWN, buff=0.35)

        self.play(Write(title), Create(ig), Create(og), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(l) for l in il], lag_ratio=0.04),
                  run_time=0.4)
        self.play(Write(formula))

        out_lbls = VGroup()
        for pi in range(2):
            for pj in range(2):
                p   = patch_rect(ic, pi*2, pj*2, 2, 2, C_POOL)
                win = vals[pi*2:pi*2+2, pj*2:pj*2+2]
                mx  = int(win.max())
                self.play(FadeIn(p), run_time=0.25)
                lbl = MathTex(str(mx), color=C_POOL, font_size=18)\
                              .move_to(oc[pi][pj])
                oc[pi][pj].set_fill(C_POOL, opacity=0.55)
                self.play(FadeIn(lbl), run_time=0.2)
                out_lbls.add(lbl)
                self.play(FadeOut(p), run_time=0.15)

        shp = MathTex(r"\text{shape: }16\times6\times6",
                      color=C_GREY, font_size=22).next_to(og, DOWN, buff=0.25)
        self.play(Write(shp))
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, ig, il, og, out_lbls, formula, shp)))

    # ── flatten ───────────────────────────────────────────────────────────────
    def _flatten_scene(self):
        title = section_title(
            r"\text{Flatten}:\ 16\times6\times6\;\Rightarrow\;\mathbb{R}^{576}",
            C_FC, math=True)

        # 3 stacked 3×3 grids (representing 3 of 16 feature maps)
        fmaps = VGroup()
        for i in range(3):
            g, _ = make_grid(3, 3, 0.30, C_FC,
                             fill_op=0.18 + i*0.1, stroke=C_FC)
            g.shift(RIGHT*i*0.16 + UP*i*0.16)
            fmaps.add(g)
        fmaps.shift(LEFT * 3.2)
        fm_lbl = MathTex(r"Y\in\mathbb{R}^{16\times6\times6}",
                         color=C_FC, font_size=22)\
                         .next_to(fmaps, DOWN, buff=0.3)

        # destination vector
        N_SHOW = 14
        dots = VGroup(*[
            Dot(radius=0.07, color=C_FC, fill_opacity=0.85)
            for _ in range(N_SHOW)
        ]).arrange(DOWN, buff=0.06).shift(RIGHT * 2.8)
        ell = MathTex(r"\vdots", color=C_GREY, font_size=22)\
                      .next_to(dots, DOWN, buff=0.05)
        vec_lbl = MathTex(r"\mathbb{R}^{576}", color=C_FC, font_size=26)\
                          .next_to(dots, RIGHT, buff=0.3)

        arr = Arrow(fmaps.get_right(), dots.get_left(),
                    color=C_ARR, buff=0.15, stroke_width=2)

        self.play(Write(title), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(g) for g in fmaps], lag_ratio=0.2),
                  run_time=0.6)
        self.play(Write(fm_lbl))
        self.wait(0.4)
        self.play(GrowArrow(arr))
        self.play(LaggedStart(*[FadeIn(d, shift=RIGHT*0.1) for d in dots],
                               lag_ratio=0.04), run_time=0.7)
        self.play(FadeIn(ell), Write(vec_lbl))
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, fmaps, fm_lbl, arr, dots, ell, vec_lbl)))

    # ── fully-connected ───────────────────────────────────────────────────────
    def _fc_scene(self):
        title = section_title(
            r"\mathbb{R}^{576}\xrightarrow{W_1}\mathbb{R}^{120}"
            r"\xrightarrow{W_2}\mathbb{R}^{84}"
            r"\xrightarrow{W_3}\mathbb{R}^{10}",
            C_FC, math=True)

        N_IN, N_OUT = 5, 4
        RAD = 0.15
        in_dots = VGroup(*[
            Circle(radius=RAD, fill_color=C_IN, fill_opacity=0.85,
                   stroke_width=0) for _ in range(N_IN)
        ]).arrange(DOWN, buff=0.3).move_to(LEFT * 3.2)

        out_dots = VGroup(*[
            Circle(radius=RAD, fill_color=C_FC, fill_opacity=0.85,
                   stroke_width=0) for _ in range(N_OUT)
        ]).arrange(DOWN, buff=0.4).move_to(RIGHT * 3.2)

        lines = VGroup(*[
            Line(d_in.get_right(), d_out.get_left(),
                 stroke_width=0.5, stroke_color=C_GREY, stroke_opacity=0.35)
            for d_in in in_dots for d_out in out_dots
        ])

        in_lbl  = MathTex(r"\mathbf{x}\in\mathbb{R}^{576}",
                          color=C_IN, font_size=22)\
                          .next_to(in_dots, DOWN, buff=0.25)
        out_lbl = MathTex(r"\mathbf{y}\in\mathbb{R}^{120}",
                          color=C_FC, font_size=22)\
                          .next_to(out_dots, DOWN, buff=0.25)
        ell_in  = MathTex(r"\vdots", color=C_GREY, font_size=20)\
                          .next_to(in_dots, UP, buff=0.1)
        ell_out = MathTex(r"\vdots", color=C_GREY, font_size=20)\
                          .next_to(out_dots, UP, buff=0.1)

        formula = MathTex(r"y_j=\sum_{i}w_{ji}\,x_i+b_j",
                          color=WHITE, font_size=28)\
                          .to_edge(DOWN, buff=0.45)

        self.play(Write(title), run_time=0.5)
        self.play(Create(lines), run_time=0.4)
        self.play(FadeIn(in_dots), FadeIn(out_dots))
        self.play(Write(in_lbl), Write(out_lbl),
                  FadeIn(ell_in), FadeIn(ell_out))
        self.play(Write(formula))
        self.wait(0.5)

        # highlight one output neuron's fan-in
        hi_lines = VGroup(*[
            Line(in_dots[i].get_right(), out_dots[1].get_left(),
                 stroke_width=2.2, stroke_color=C_CONV)
            for i in range(N_IN)
        ])
        hi_dot = out_dots[1].copy().set_fill(WHITE, opacity=0.9)
        detail = MathTex(
            r"y_1=w_{10}x_0+\cdots+w_{1,575}x_{575}+b_1",
            color=C_CONV, font_size=16)\
            .next_to(formula, UP, buff=0.2)

        self.play(LaggedStart(*[Create(l) for l in hi_lines], lag_ratio=0.1),
                  run_time=0.7)
        self.play(Transform(out_dots[1], hi_dot), FadeIn(detail))
        self.wait(1.0)

        seq = MathTex(r"576\to120\to84\to10",
                      color=C_GREY, font_size=26)\
                      .next_to(title, DOWN, buff=0.15)
        self.play(Write(seq))
        self.wait(1.0)
        self.play(FadeOut(VGroup(
            title, lines, in_dots, out_dots,
            in_lbl, out_lbl, ell_in, ell_out,
            formula, hi_lines, detail, seq)))

    # ── full pipeline diagram ─────────────────────────────────────────────────
    def _pipeline_scene(self):
        title = Text("Full Pipeline  —  LeNet on MNIST 32×32",
                     color=WHITE, font_size=28).to_edge(UP, buff=0.4)

        specs = [
            ("Input",    C_IN,   r"1\!\times\!32\!\times\!32"),
            ("Conv1",    C_CONV, r"6\!\times\!30\!\times\!30"),
            ("ReLU",     C_RELU, r"6\!\times\!30\!\times\!30"),
            ("Pool1",    C_POOL, r"6\!\times\!15\!\times\!15"),
            ("Conv2",    C_CONV, r"16\!\times\!12\!\times\!12"),
            ("ReLU",     C_RELU, r"16\!\times\!12\!\times\!12"),
            ("Pool2",    C_POOL, r"16\!\times\!6\!\times\!6"),
            ("Flatten",  C_FC,   r"576"),
            ("FC1",      C_FC,   r"120"),
            ("FC2",      C_FC,   r"84"),
            ("Output",   C_OUT,  r"10"),
        ]

        entries = VGroup()
        for name, col, shp in specs:
            lbl     = Text(name, color=col, font_size=14)
            shp_tex = MathTex(shp, color=C_GREY, font_size=11)
            content = VGroup(lbl, shp_tex).arrange(DOWN, buff=0.05)
            rect    = SurroundingRectangle(content, color=col,
                                           buff=0.1, corner_radius=0.07)
            entries.add(VGroup(rect, content))

        entries.arrange(RIGHT, buff=0.14).center().shift(DOWN * 0.1)

        arrows = VGroup(*[
            Arrow(entries[i].get_right(), entries[i+1].get_left(),
                  buff=0.04, color=C_ARR, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.3)
            for i in range(len(entries)-1)
        ])

        self.play(Write(title))
        self.play(LaggedStart(*[FadeIn(e) for e in entries],
                               lag_ratio=0.1), run_time=2.0)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows],
                               lag_ratio=0.08), run_time=1.5)
        self.wait(3.0)
