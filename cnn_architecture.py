"""
CNN Teaching Animation  —  Manim Community v0.20.1

Explains visually how an MNIST digit "7" passes through two Conv2D layers:

  SCENE 1  Title
  SCENE 2  Input 1×32×32  → Conv1 (Ci=1, Co=2, k=3)  → Y1 2×30×30
  SCENE 3  Transition: Y1 becomes new input for layer 2
  SCENE 4  Conv2 (Ci=2, Co=4, k=3): 4×2 filter bank, detailed for ch-0
  SCENE 5  Build the 4 output feature maps Y2 4×28×28
  SCENE 6  Parameter summary
"""

from manim import *
import numpy as np

# ── palette ──────────────────────────────────────────────────────────────────
BG     = BLACK
C_IN   = "#4FC3F7"   # blue   – raw input / feature map 0
C_FM1  = "#FF9800"   # orange – feature map 1 (Y1 channel 1)
C_K1   = "#FFD54F"   # amber  – kernels layer 1
C_OUT0 = "#A5D6A7"   # green  – output channel 0
C_OUT1 = "#CE93D8"   # purple – output channel 1
C_OUT2 = "#EF9A9A"   # red    – output channel 2
C_OUT3 = "#80DEEA"   # cyan   – output channel 3
C_GREY = "#B0BEC5"
C_ARR  = "#90A4AE"

OUT_COLORS = [C_OUT0, C_OUT1, C_OUT2, C_OUT3]
OUT_NAMES  = ["o=0", "o=1", "o=2", "o=3"]

rng = np.random.default_rng(42)

# ── stylised 32×32 digit "7" (numpy array, values 0/1) ───────────────────────
def make_digit7():
    d = np.zeros((32, 32), dtype=float)
    # horizontal top bar  rows 4-7, cols 6-25
    d[4:8,  6:26] = 1.0
    # diagonal stem       rows 8-27
    for i in range(20):
        col_start = 17 + i // 2
        col_end   = min(col_start + 4, 32)
        d[8 + i, col_start:col_end] = 1.0
    return d


# ── helpers ──────────────────────────────────────────────────────────────────

def make_grid(rows, cols, cs=0.40,
              fill=C_IN, fill_op=0.15,
              stroke=WHITE, sw=0.7):
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


def paint_digit(cells, digit, color=C_IN):
    """Set fill opacity of cells proportional to digit pixel values."""
    anims = []
    for r, row in enumerate(cells):
        for c, sq in enumerate(row):
            op = float(digit[r, c]) * 0.85 + 0.05
            anims.append(sq.animate.set_fill(color, opacity=op))
    return anims


def patch_rect(cells, r0, c0, kr, kc, color=C_K1, sw=2.5, fill_op=0.20):
    """Highlight rectangle over cells[r0:r0+kr, c0:c0+kc]."""
    tl = cells[r0][c0].get_corner(UL)
    br = cells[r0 + kr - 1][c0 + kc - 1].get_corner(DR)
    w, h = br[0] - tl[0], tl[1] - br[1]
    rect = Rectangle(width=w, height=h,
                     stroke_color=color, stroke_width=sw,
                     fill_color=color, fill_opacity=fill_op)
    rect.move_to((tl + br) / 2)
    return rect


def labeled_map(name_tex, size_tex, color, cs=0.38, n=6):
    """Small n×n feature-map tile with a label."""
    grp, cells = make_grid(n, n, cs=cs, fill=color,
                           fill_op=0.18, stroke=color, sw=0.8)
    # random-ish brightness to suggest content
    for row in cells:
        for sq in row:
            sq.set_fill(opacity=rng.uniform(0.1, 0.7))
    lbl = MathTex(name_tex, color=color, font_size=22)
    sz  = MathTex(size_tex, color=C_GREY, font_size=18)
    lbl.next_to(grp, UP, buff=0.12)
    sz.next_to(grp, DOWN, buff=0.10)
    return VGroup(lbl, grp, sz), cells


def kernel_tile(name_tex, color, cs=0.34):
    """3×3 kernel tile with label."""
    grp, cells = make_grid(3, 3, cs=cs, fill=color,
                           fill_op=0.25, stroke=color, sw=1.2)
    for row in cells:
        for sq in row:
            sq.set_fill(opacity=rng.uniform(0.2, 0.8))
    lbl = MathTex(name_tex, color=color, font_size=20)
    lbl.next_to(grp, UP, buff=0.10)
    return VGroup(lbl, grp), cells


def param_box(lines, colors=None):
    """Bordered box with lines of MathTex."""
    if colors is None:
        colors = [WHITE] * len(lines)
    texs = VGroup(*[
        MathTex(l, color=c, font_size=22)
        for l, c in zip(lines, colors)
    ]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    box = SurroundingRectangle(texs, color=C_GREY,
                               buff=0.25, corner_radius=0.12)
    return VGroup(box, texs)


# ── main scene ────────────────────────────────────────────────────────────────

class CNNArchitecture(Scene):

    def construct(self):
        self.camera.background_color = BG
        self._scene1_title()
        self._scene2_conv1()
        self._scene3_transition()
        self._scene4_conv2_detail()
        self._scene5_output_maps()
        self._scene6_summary()

    # ── SCENE 1 — Title ───────────────────────────────────────────────────────
    def _scene1_title(self):
        t = Text(
            "Conv2D on MNIST: from input channels to output channels",
            color=WHITE, font_size=32)
        sub = MathTex(
            r"\text{Layer 1: }C_i=1,\;C_o=2"
            r"\quad|\quad"
            r"\text{Layer 2: }C_i=2,\;C_o=4",
            color=C_GREY, font_size=24).next_to(t, DOWN, buff=0.3)
        self.play(Write(t), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.wait(2.0)
        self.play(FadeOut(VGroup(t, sub)))

    # ── SCENE 2 — Input digit + Conv1 ─────────────────────────────────────────
    def _scene2_conv1(self):
        # ---- title -----------------------------------------------------------
        title = MathTex(
            r"\text{Layer 1:}\quad X\in\mathbb{R}^{1\times32\times32}"
            r"\;\xrightarrow{W_1\in\mathbb{R}^{2\times1\times3\times3}}\;"
            r"Y_1\in\mathbb{R}^{2\times30\times30}",
            color=WHITE, font_size=22).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # ---- input grid (8×8 excerpt of 32×32 digit 7) -----------------------
        digit = make_digit7()
        excerpt_n = 8
        excerpt   = digit[:excerpt_n, :excerpt_n]

        CS_IN = 0.36
        inp_grp, inp_cells = make_grid(
            excerpt_n, excerpt_n, cs=CS_IN,
            fill=C_IN, fill_op=0.15, stroke=C_IN, sw=0.6)
        inp_grp.shift(LEFT * 3.8)

        inp_lbl  = MathTex(r"X\in\mathbb{R}^{1\times32\times32}",
                           color=C_IN, font_size=22)
        inp_lbl.next_to(inp_grp, UP, buff=0.15)
        inp_note = MathTex(r"(8\times8\text{ excerpt})",
                           color=C_GREY, font_size=16)
        inp_note.next_to(inp_grp, DOWN, buff=0.10)
        ch_note  = MathTex(r"C_{in}=1", color=C_GREY, font_size=18)
        ch_note.next_to(inp_note, DOWN, buff=0.05)

        self.play(Create(inp_grp), run_time=0.8)
        # paint digit pixels
        paint_anims = paint_digit(inp_cells, excerpt, C_IN)
        self.play(*paint_anims, run_time=0.8)
        self.play(Write(inp_lbl), Write(inp_note), Write(ch_note))
        self.wait(0.5)

        # ---- two kernels K1, K2 (centre) ------------------------------------
        CS_K = 0.32
        k1_grp, k1_cells = make_grid(
            3, 3, cs=CS_K, fill=C_K1, fill_op=0.25, stroke=C_K1, sw=1.2)
        k2_grp, k2_cells = make_grid(
            3, 3, cs=CS_K, fill=C_FM1, fill_op=0.25, stroke=C_FM1, sw=1.2)

        # fixed random values for display
        k1_vals = np.array([[ 1, 0,-1],[ 1, 0,-1],[ 1, 0,-1]], dtype=float)
        k2_vals = np.array([[ 1, 1, 1],[ 0, 0, 0],[-1,-1,-1]], dtype=float)
        for r in range(3):
            for c in range(3):
                k1_cells[r][c].set_fill(opacity=abs(k1_vals[r,c]) * 0.5 + 0.15)
                k2_cells[r][c].set_fill(opacity=abs(k2_vals[r,c]) * 0.5 + 0.15)

        k1_lbl = MathTex(r"K_1", color=C_K1, font_size=22)
        k2_lbl = MathTex(r"K_2", color=C_FM1, font_size=22)
        k1_lbl.next_to(k1_grp, UP, buff=0.10)
        k2_lbl.next_to(k2_grp, UP, buff=0.10)

        k1_group = VGroup(k1_lbl, k1_grp)
        k2_group = VGroup(k2_lbl, k2_grp)
        VGroup(k1_group, k2_group).arrange(DOWN, buff=0.4).move_to(ORIGIN)

        self.play(FadeIn(k1_group), FadeIn(k2_group))
        self.wait(0.4)

        # ---- two output feature maps (right) --------------------------------
        FM_N  = 6    # show 6×6 excerpt of 30×30 output
        CS_FM = 0.36

        fm1_grp, fm1_cells = make_grid(
            FM_N, FM_N, cs=CS_FM, fill=C_K1,
            fill_op=0.08, stroke=C_K1, sw=0.7)
        fm2_grp, fm2_cells = make_grid(
            FM_N, FM_N, cs=CS_FM, fill=C_FM1,
            fill_op=0.08, stroke=C_FM1, sw=0.7)

        fm1_lbl  = MathTex(r"Y_1[0,\cdot,\cdot]", color=C_K1, font_size=20)
        fm2_lbl  = MathTex(r"Y_1[1,\cdot,\cdot]", color=C_FM1, font_size=20)
        fm1_lbl.next_to(fm1_grp, UP, buff=0.12)
        fm2_lbl.next_to(fm2_grp, UP, buff=0.12)

        fm1_group = VGroup(fm1_lbl, fm1_grp)
        fm2_group = VGroup(fm2_lbl, fm2_grp)
        VGroup(fm1_group, fm2_group).arrange(DOWN, buff=0.5).shift(RIGHT * 3.8)

        self.play(Create(fm1_grp), Create(fm2_grp),
                  Write(fm1_lbl), Write(fm2_lbl))
        self.wait(0.3)

        # ---- formula at bottom -----------------------------------------------
        formula = MathTex(
            r"Y_1[o,i,j]=b_o+"
            r"\sum_{u}\sum_{v}X[0,i{+}u,j{+}v]\cdot W_1[o,0,u,v]"
            r"\quad(c=0\text{ only})",
            color=WHITE, font_size=18).to_edge(DOWN, buff=0.25)
        self.play(Write(formula))

        # ---- animate sliding K1 on input → fills fm1 -------------------------
        patch = None
        positions = [(0,0),(0,2),(1,1),(2,0),(2,2),(3,3),(4,1),(5,4)]
        for idx, (r0, c0) in enumerate(positions):
            # clamp to valid positions for 8×8 input, 3×3 kernel
            r0 = min(r0, excerpt_n - 3)
            c0 = min(c0, excerpt_n - 3)
            new_p = patch_rect(inp_cells, r0, c0, 3, 3, color=C_K1)
            # move kernel visual to patch position
            patch_center = new_p.get_center()
            k1_anim = k1_grp.animate.move_to(
                patch_center + RIGHT * 0.0)  # overlay on patch

            if patch is None:
                patch = new_p
                self.play(FadeIn(patch), k1_anim,
                          run_time=0.4)
            else:
                self.play(Transform(patch, new_p), k1_anim,
                          run_time=0.25 if idx < 3 else 0.15)

            # light up corresponding output cell (map to 6×6 grid)
            or_, oc_ = min(r0, FM_N - 1), min(c0, FM_N - 1)
            fm1_cells[or_][oc_].set_fill(C_K1, opacity=0.6)
            rt = 0.2 if idx < 3 else 0.1
            self.play(Flash(fm1_cells[or_][oc_], color=C_K1,
                            flash_radius=CS_FM * 0.65,
                            line_length=CS_FM * 0.3),
                      run_time=rt)
            if idx < 2:
                self.wait(0.2)

        # fill remaining output cells schematically
        fill_anims = []
        for row in fm1_cells:
            for sq in row:
                if sq.get_fill_opacity() < 0.3:
                    fill_anims.append(
                        sq.animate.set_fill(C_K1, opacity=rng.uniform(0.25, 0.7)))
        self.play(*fill_anims, run_time=0.5)

        # reset k1 position
        self.play(k1_grp.animate.move_to(k1_group.get_center()),
                  FadeOut(patch), run_time=0.3)

        # ---- animate sliding K2 on input → fills fm2 -------------------------
        patch2 = None
        for idx, (r0, c0) in enumerate([(0,1),(1,3),(2,2),(3,0),(4,4),(5,2)]):
            r0 = min(r0, excerpt_n - 3)
            c0 = min(c0, excerpt_n - 3)
            new_p = patch_rect(inp_cells, r0, c0, 3, 3, color=C_FM1)
            k2_anim = k2_grp.animate.move_to(new_p.get_center())

            if patch2 is None:
                patch2 = new_p
                self.play(FadeIn(patch2), k2_anim, run_time=0.35)
            else:
                self.play(Transform(patch2, new_p), k2_anim,
                          run_time=0.20 if idx < 3 else 0.12)

            or_, oc_ = min(r0, FM_N - 1), min(c0, FM_N - 1)
            fm2_cells[or_][oc_].set_fill(C_FM1, opacity=0.6)
            self.play(Flash(fm2_cells[or_][oc_], color=C_FM1,
                            flash_radius=CS_FM * 0.65,
                            line_length=CS_FM * 0.3),
                      run_time=0.12)

        fill_anims2 = []
        for row in fm2_cells:
            for sq in row:
                if sq.get_fill_opacity() < 0.3:
                    fill_anims2.append(
                        sq.animate.set_fill(C_FM1, opacity=rng.uniform(0.25, 0.7)))
        self.play(*fill_anims2, run_time=0.5)
        self.play(k2_grp.animate.move_to(k2_group.get_center()),
                  FadeOut(patch2), run_time=0.3)

        # ---- layer-1 summary annotation -------------------------------------
        summary = MathTex(
            r"W_1\in\mathbb{R}^{2\times1\times3\times3}\quad "
            r"Y_1\in\mathbb{R}^{2\times30\times30}\quad "
            r"\Omega_1=20",
            color=C_GREY, font_size=18).next_to(formula, UP, buff=0.15)
        self.play(FadeIn(summary))
        self.wait(2.0)

        # store references for scene 3 transition
        self._fm1_group  = fm1_group
        self._fm2_group  = fm2_group
        self._scene2_all = VGroup(
            title, inp_grp, inp_lbl, inp_note, ch_note,
            k1_group, k2_group, formula, summary)

    # ── SCENE 3 — Transition: Y1 becomes new input ────────────────────────────
    def _scene3_transition(self):
        title3 = MathTex(
            r"\text{Transition: }Y_1\text{ becomes the input to Layer 2}",
            color=WHITE, font_size=26).to_edge(UP, buff=0.3)

        # fade out everything except the two feature maps
        self.play(FadeOut(self._scene2_all), Write(title3))
        self.wait(0.5)

        # slide the two feature maps to the left side
        target_pos = LEFT * 3.4
        fm1_target = self._fm1_group.copy()
        fm2_target = self._fm2_group.copy()
        VGroup(fm1_target, fm2_target).arrange(DOWN, buff=0.5).move_to(target_pos)

        self.play(
            self._fm1_group.animate.move_to(fm1_target.get_center()),
            self._fm2_group.animate.move_to(fm2_target.get_center()),
            run_time=1.0)

        new_inp_lbl = MathTex(
            r"\text{New input to Layer 2}",
            color=WHITE, font_size=20).next_to(
                VGroup(self._fm1_group, self._fm2_group), UP, buff=0.2)
        y1_lbl = MathTex(r"Y_1\in\mathbb{R}^{2\times30\times30}",
                         color=C_GREY, font_size=18).next_to(
                             VGroup(self._fm1_group, self._fm2_group), DOWN, buff=0.1)
        ci_lbl = MathTex(r"C_i=2", color=C_GREY, font_size=20).next_to(
            y1_lbl, DOWN, buff=0.05)

        self.play(Write(new_inp_lbl), Write(y1_lbl), Write(ci_lbl))
        self.wait(2.0)

        self._trans_title   = title3
        self._trans_inp_lbl = new_inp_lbl
        self._trans_y1_lbl  = y1_lbl
        self._trans_ci_lbl  = ci_lbl

    # ── SCENE 4 — Conv2: detailed for output ch 0, compact for ch 1-3 ─────────
    def _scene4_conv2_detail(self):
        title4 = MathTex(
            r"\text{Layer 2:}\quad Y_1\in\mathbb{R}^{2\times30\times30}"
            r"\;\xrightarrow{W_2\in\mathbb{R}^{4\times2\times3\times3}}\;"
            r"Y_2\in\mathbb{R}^{4\times28\times28}",
            color=WHITE, font_size=20).to_edge(UP, buff=0.3)

        self.play(
            FadeOut(VGroup(self._trans_title, self._trans_inp_lbl,
                           self._trans_y1_lbl, self._trans_ci_lbl)),
            Write(title4))

        # ---- 4×2 kernel bank diagram (centre) --------------------------------
        bank_title = MathTex(r"W_2\in\mathbb{R}^{4\times2\times3\times3}",
                             color=WHITE, font_size=20).shift(UP * 1.6)

        row_labels  = [MathTex(rf"o={i}", color=OUT_COLORS[i], font_size=16)
                       for i in range(4)]
        col_labels  = [MathTex(rf"c={j}", color=[C_IN, C_FM1][j], font_size=16)
                       for j in range(2)]

        CS_K = 0.26
        bank_cells = []   # [out][in] → (VGroup tile, cells)
        bank_group = VGroup()

        for o in range(4):
            row_group = VGroup()
            row_tiles = []
            for ci in range(2):
                color = OUT_COLORS[o]
                inp_c = [C_IN, C_FM1][ci]
                g, cells_ = make_grid(3, 3, cs=CS_K,
                                      fill=color, fill_op=0.20,
                                      stroke=color, sw=0.9)
                for row in cells_:
                    for sq in row:
                        sq.set_fill(opacity=rng.uniform(0.15, 0.75))
                row_group.add(g)
                row_tiles.append((g, cells_))
            row_group.arrange(RIGHT, buff=0.25)
            bank_group.add(row_group)
            bank_cells.append(row_tiles)

        bank_group.arrange(DOWN, buff=0.30).move_to(RIGHT * 0.8 + DOWN * 0.2)

        # column headers
        col_h = VGroup(*[
            MathTex(rf"c={j}", color=[C_IN, C_FM1][j], font_size=14)
            for j in range(2)
        ])
        for j, (grp, _) in enumerate(bank_cells[0]):
            col_h[j].next_to(grp, UP, buff=0.12)

        # row headers
        row_h = VGroup(*[
            MathTex(rf"o={i}", color=OUT_COLORS[i], font_size=14)
            for i in range(4)
        ])
        for i in range(4):
            row_h[i].next_to(bank_cells[i][0][0], LEFT, buff=0.15)

        self.play(Write(bank_title))
        self.play(FadeIn(bank_group), FadeIn(col_h), FadeIn(row_h))
        self.wait(0.5)

        # ---- detailed animation for output channel o=0 ----------------------
        detail_lbl = MathTex(
            r"\text{Output channel }o=0\text{ in detail:}",
            color=OUT_COLORS[0], font_size=18).to_edge(DOWN, buff=1.0)
        self.play(Write(detail_lbl))

        # highlight row 0 of bank
        self.play(
            bank_cells[0][0][0].animate.set_stroke(color=C_IN,     width=2.5),
            bank_cells[0][1][0].animate.set_stroke(color=C_FM1,    width=2.5))

        # show the two input feature maps on the left still in place
        fm1 = self._fm1_group
        fm2 = self._fm2_group

        # get the 6×6 grid cells from the stored groups (index 1 = grid VGroup)
        fm1_grid = fm1[1]
        fm2_grid = fm2[1]

        # animate patch on fm1 for kernel W[0,0]
        FM_N = 6
        fm1_cells_ref = [[fm1_grid[r * FM_N + c] for c in range(FM_N)]
                         for r in range(FM_N)]
        fm2_cells_ref = [[fm2_grid[r * FM_N + c] for c in range(FM_N)]
                         for r in range(FM_N)]

        def do_brief_slide(cells_ref, color, n_steps=4):
            patch_ = None
            positions_ = [(0,0),(0,2),(1,1),(2,3)][:n_steps]
            for idx_, (r0, c0) in enumerate(positions_):
                r0 = min(r0, FM_N - 3)
                c0 = min(c0, FM_N - 3)
                np_ = patch_rect(cells_ref, r0, c0, 3, 3,
                                 color=color, sw=2.2)
                if patch_ is None:
                    patch_ = np_
                    self.play(FadeIn(patch_), run_time=0.3)
                else:
                    self.play(Transform(patch_, np_), run_time=0.18)
            return patch_

        patch_on_fm1 = do_brief_slide(fm1_cells_ref, C_IN,    n_steps=3)
        patch_on_fm2 = do_brief_slide(fm2_cells_ref, C_FM1,   n_steps=3)

        # show summation formula
        sum_formula = MathTex(
            r"Y_2[0,i,j]=b_0+"
            r"\underbrace{\text{conv}(Y_1[0],\,W_2[0,0])}_{\text{ch 0}}"
            r"+\underbrace{\text{conv}(Y_1[1],\,W_2[0,1])}_{\text{ch 1}}",
            color=WHITE, font_size=16).to_edge(DOWN, buff=0.25)
        self.play(Write(sum_formula))
        self.wait(1.5)

        self.play(FadeOut(patch_on_fm1), FadeOut(patch_on_fm2),
                  FadeOut(detail_lbl))

        # ---- compact labels for ch 1-3 ---------------------------------------
        compact_lines = VGroup()
        for o in range(1, 4):
            line = MathTex(
                rf"Y_2[{o},i,j]=b_{o}+"
                rf"\sum_{{c=0}}^{{1}}"
                rf"\text{{conv}}(Y_1[c],\,W_2[{o},c])",
                color=OUT_COLORS[o], font_size=14)
            compact_lines.add(line)
        compact_lines.arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.25)

        self.play(FadeOut(sum_formula))
        self.play(LaggedStart(*[FadeIn(l) for l in compact_lines],
                               lag_ratio=0.3), run_time=1.2)
        self.wait(1.5)

        self._title4     = title4
        self._bank_group = bank_group
        self._col_h      = col_h
        self._row_h      = row_h
        self._bank_title = bank_title
        self._compact    = compact_lines

    # ── SCENE 5 — Build the 4 output feature maps ─────────────────────────────
    def _scene5_output_maps(self):
        title5 = MathTex(
            r"Y_2\in\mathbb{R}^{4\times28\times28}\quad\Omega_2=76",
            color=WHITE, font_size=26).to_edge(UP, buff=0.3)

        self.play(
            FadeOut(VGroup(self._title4, self._bank_group, self._col_h,
                           self._row_h, self._bank_title, self._compact,
                           self._fm1_group, self._fm2_group)),
            Write(title5))

        FM_N  = 5
        CS_FM = 0.38
        out_groups = []
        for o in range(4):
            g, cells_ = make_grid(FM_N, FM_N, cs=CS_FM,
                                  fill=OUT_COLORS[o],
                                  fill_op=0.08, stroke=OUT_COLORS[o], sw=0.8)
            lbl = MathTex(rf"Y_2[{o},\cdot,\cdot]",
                          color=OUT_COLORS[o], font_size=18)
            lbl.next_to(g, UP, buff=0.12)
            sz = MathTex(r"28\times28", color=C_GREY, font_size=14)
            sz.next_to(g, DOWN, buff=0.08)
            out_groups.append(VGroup(lbl, g, sz))

        row_top = VGroup(out_groups[0], out_groups[1]).arrange(RIGHT, buff=0.6)
        row_bot = VGroup(out_groups[2], out_groups[3]).arrange(RIGHT, buff=0.6)
        VGroup(row_top, row_bot).arrange(DOWN, buff=0.5).center().shift(DOWN * 0.2)

        # appear one by one, filling cells as they appear
        for o, og in enumerate(out_groups):
            g_ref = og[1]   # the grid VGroup
            fill_anims = [
                sq.animate.set_fill(OUT_COLORS[o],
                                    opacity=rng.uniform(0.2, 0.75))
                for sq in g_ref
            ]
            self.play(FadeIn(og[0]), Create(g_ref), FadeIn(og[2]),
                      run_time=0.6)
            self.play(*fill_anims, run_time=0.5)
            self.wait(0.3)

        self.wait(1.5)

        self._title5    = title5
        self._out_group = VGroup(*out_groups)

    # ── SCENE 6 — Parameter summary ───────────────────────────────────────────
    def _scene6_summary(self):
        self.play(FadeOut(VGroup(self._title5, self._out_group)))

        title6 = Text("Parameter Summary", color=WHITE, font_size=32)\
                     .to_edge(UP, buff=0.35)
        self.play(Write(title6))

        L1_lines = [
            r"X\in\mathbb{R}^{1\times32\times32}",
            r"W_1\in\mathbb{R}^{2\times1\times3\times3}",
            r"Y_1\in\mathbb{R}^{2\times30\times30}",
            r"\Omega_1=2\times(1\times3\times3+1)=20",
        ]
        L2_lines = [
            r"Y_1\in\mathbb{R}^{2\times30\times30}",
            r"W_2\in\mathbb{R}^{4\times2\times3\times3}",
            r"Y_2\in\mathbb{R}^{4\times28\times28}",
            r"\Omega_2=4\times(2\times3\times3+1)=76",
        ]

        L1_colors = [C_IN,   C_K1,  C_K1,  C_GREY]
        L2_colors = [C_FM1,  C_OUT0, C_OUT0, C_GREY]

        box1 = param_box([r"\text{Layer 1}:"] + L1_lines,
                         colors=[WHITE] + L1_colors)
        box2 = param_box([r"\text{Layer 2}:"] + L2_lines,
                         colors=[WHITE] + L2_colors)

        VGroup(box1, box2).arrange(RIGHT, buff=0.7).center().shift(DOWN * 0.2)

        total = MathTex(
            r"|\Omega|=\Omega_1+\Omega_2=20+76=\mathbf{96}",
            color=WHITE, font_size=26).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(box1), FadeIn(box2))
        self.wait(0.8)
        self.play(Write(total))
        self.wait(3.0)
