from manim import *
import numpy as np

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG      = BLACK
XCOLOR  = "#4FC3F7"   # light blue  – input / feature maps
KCOLOR  = "#FFD54F"   # amber       – kernels
YCOLOR  = "#A5D6A7"   # light green – outputs
EQCOLOR = WHITE


# ---------------------------------------------------------------------------
# Helper: draw a grid of cells with optional labels
# ---------------------------------------------------------------------------
def make_grid(rows, cols, cell_size=0.38, label_fn=None,
              fill_color=XCOLOR, fill_opacity=0.15, stroke_color=WHITE,
              stroke_width=0.8):
    """Return a VGroup with the grid cells (and optional text labels)."""
    grid = VGroup()
    for r in range(rows):
        for c in range(cols):
            rect = Square(side_length=cell_size,
                          fill_color=fill_color,
                          fill_opacity=fill_opacity,
                          stroke_color=stroke_color,
                          stroke_width=stroke_width)
            rect.move_to(RIGHT * c * cell_size + DOWN * r * cell_size)
            if label_fn:
                lbl = label_fn(r, c).scale(0.28).move_to(rect)
                grid.add(rect, lbl)
            else:
                grid.add(rect)
    grid.center()
    return grid


# ---------------------------------------------------------------------------
# Helper: overlay a highlight rectangle on a subgrid
# ---------------------------------------------------------------------------
def highlight_patch(grid_topleft, r0, c0, ksize, cell_size=0.38):
    side = cell_size * ksize
    rect = Rectangle(width=side, height=side,
                     stroke_color=KCOLOR, stroke_width=2.5,
                     fill_color=KCOLOR, fill_opacity=0.15)
    cx = grid_topleft[0] + (c0 + ksize / 2 - 0.5) * cell_size
    cy = grid_topleft[1] - (r0 + ksize / 2 - 0.5) * cell_size
    rect.move_to([cx, cy, 0])
    return rect


# ===========================================================================
class Conv2DExplanation(Scene):
    """
    Minimalist, mathematically rigorous explanation of Conv2D through two
    successive layers, starting from a single-channel 32×32 MNIST digit.
    """

    # -----------------------------------------------------------------------
    def construct(self):
        self.camera.background_color = BG

        self._scene_title()
        self._scene_input()
        self._scene_layer1_formula()
        self._scene_layer1_slide()
        self._scene_layer2_formula()
        self._scene_summary()

    # -----------------------------------------------------------------------
    # Scene 0 – title
    # -----------------------------------------------------------------------
    def _scene_title(self):
        t1 = Tex(r"Conv2D $\,\cdot\,$ Two-Layer Walk-Through",
                 color=WHITE).scale(0.9)
        t2 = MathTex(r"X \in \mathbb{R}^{1 \times 32 \times 32}",
                     color=XCOLOR).scale(0.7).next_to(t1, DOWN, buff=0.3)
        self.play(Write(t1), run_time=1)
        self.play(FadeIn(t2, shift=UP * 0.2))
        self.wait(1)
        self.play(FadeOut(VGroup(t1, t2)))

    # -----------------------------------------------------------------------
    # Scene 1 – draw input grid
    # -----------------------------------------------------------------------
    def _scene_input(self):
        N = 7   # display 7×7 excerpt of the 32×32 input

        title = MathTex(r"X \in \mathbb{R}^{1 \times 32 \times 32}",
                        color=XCOLOR).scale(0.75).to_edge(UP, buff=0.4)

        grid = make_grid(
            N, N, cell_size=0.46,
            label_fn=lambda r, c: MathTex(
                rf"x_{{{r},{c}}}", color=XCOLOR),
            fill_color=XCOLOR)
        grid.shift(LEFT * 0.3)

        brace_h = Brace(grid, direction=DOWN, color=GRAY)
        brace_v = Brace(grid, direction=LEFT,  color=GRAY)
        lbl_h = brace_h.get_tex(r"W=32").set_color(GRAY).scale(0.7)
        lbl_v = brace_v.get_tex(r"H=32").set_color(GRAY).scale(0.7)

        channel_lbl = MathTex(r"C_{in}=1", color=GRAY).scale(0.65)\
                              .next_to(title, RIGHT, buff=0.6)

        self.play(Write(title))
        self.play(Create(grid), run_time=1.5)
        self.play(GrowFromCenter(brace_h), Write(lbl_h),
                  GrowFromCenter(brace_v), Write(lbl_v),
                  FadeIn(channel_lbl))
        self.wait(1.2)
        self.play(FadeOut(VGroup(grid, brace_h, brace_v,
                                 lbl_h, lbl_v, title, channel_lbl)))

    # -----------------------------------------------------------------------
    # Scene 2 – Layer-1 mathematical formula
    # -----------------------------------------------------------------------
    def _scene_layer1_formula(self):
        head = Tex(r"\textbf{Layer 1}", color=WHITE).scale(0.85).to_edge(UP, buff=0.35)

        tensors = MathTex(
            r"X \in \mathbb{R}^{1 \times 32 \times 32}",
            r"\quad",
            r"K^{(1)} \in \mathbb{R}^{C_1 \times 1 \times k \times k}",
            color=WHITE).scale(0.65)
        tensors[0].set_color(XCOLOR)
        tensors[2].set_color(KCOLOR)
        tensors.next_to(head, DOWN, buff=0.35)

        formula = MathTex(
            r"Y_1[c,\,i,\,j]",
            r"=",
            r"\sum_{m=0}^{k-1}\sum_{n=0}^{k-1}",
            r"X[0,\;i{+}m,\;j{+}n]",
            r"\cdot",
            r"K^{(1)}[c,\,0,\,m,\,n]",
            color=WHITE).scale(0.68)
        formula[0].set_color(YCOLOR)
        formula[3].set_color(XCOLOR)
        formula[5].set_color(KCOLOR)
        formula.next_to(tensors, DOWN, buff=0.45)

        dim = MathTex(
            r"Y_1 \in \mathbb{R}^{C_1 \times H_1 \times W_1}",
            color=YCOLOR).scale(0.65).next_to(formula, DOWN, buff=0.4)

        note = MathTex(
            r"H_1 = H - k + 1,\quad W_1 = W - k + 1\quad"
            r"(\text{valid padding})",
            color=GRAY).scale(0.55).next_to(dim, DOWN, buff=0.25)

        self.play(Write(head))
        self.play(FadeIn(tensors, shift=UP * 0.15))
        self.play(Write(formula), run_time=2)
        self.play(FadeIn(dim), FadeIn(note))
        self.wait(2)
        self.play(FadeOut(VGroup(head, tensors, formula, dim, note)))

    # -----------------------------------------------------------------------
    # Scene 3 – animate the sliding kernel on a small excerpt
    # -----------------------------------------------------------------------
    def _scene_layer1_slide(self):
        G   = 5      # 5×5 input excerpt
        K   = 3      # 3×3 kernel
        CS  = 0.44   # cell size
        OUT = G - K + 1  # 3×3 output

        # ---- input grid (left) -----------------------------------------
        inp_lbl = MathTex(r"X[\,0,\,\cdot\,,\,\cdot\,]",
                          color=XCOLOR).scale(0.6)
        inp_grid = make_grid(G, G, cell_size=CS,
                             label_fn=lambda r, c: MathTex(
                                 rf"x_{{{r}{c}}}", color=XCOLOR),
                             fill_color=XCOLOR)
        inp_group = VGroup(inp_lbl, inp_grid).arrange(DOWN, buff=0.2)
        inp_group.shift(LEFT * 3.2)

        # ---- kernel grid (center) --------------------------------------
        k_lbl = MathTex(r"K^{(1)}[c,\,0,\,\cdot\,,\,\cdot\,]",
                        color=KCOLOR).scale(0.6)
        k_grid = make_grid(K, K, cell_size=CS,
                           label_fn=lambda r, c: MathTex(
                               rf"k_{{{r}{c}}}", color=KCOLOR),
                           fill_color=KCOLOR, stroke_color=KCOLOR)
        k_group = VGroup(k_lbl, k_grid).arrange(DOWN, buff=0.2)
        k_group.move_to(ORIGIN)

        # ---- output grid (right) ----------------------------------------
        out_lbl = MathTex(r"Y_1[c,\,\cdot\,,\,\cdot\,]",
                          color=YCOLOR).scale(0.6)
        out_grid = make_grid(OUT, OUT, cell_size=CS,
                             fill_color=YCOLOR,
                             stroke_color=YCOLOR)
        out_group = VGroup(out_lbl, out_grid).arrange(DOWN, buff=0.2)
        out_group.shift(RIGHT * 3.2)

        # star symbols
        star1 = MathTex(r"\ast", color=WHITE).scale(1.0).move_to(
            (inp_group.get_right() + k_group.get_left()) / 2)
        eq    = MathTex(r"=",    color=WHITE).scale(1.0).move_to(
            (k_group.get_right() + out_group.get_left()) / 2)

        self.play(FadeIn(inp_group), FadeIn(k_group),
                  FadeIn(out_group), Write(star1), Write(eq))
        self.wait(0.5)

        # dot-product formula at bottom
        dot_eq = MathTex(
            r"Y_1[c,i,j]=\sum_{m,n}",
            r"x_{i{+}m,\,j{+}n}",
            r"\cdot",
            r"k_{m,n}",
            color=WHITE).scale(0.62).to_edge(DOWN, buff=0.35)
        dot_eq[1].set_color(XCOLOR)
        dot_eq[3].set_color(KCOLOR)
        self.play(Write(dot_eq))

        # ---- animate sliding patch on input ----------------------------
        inp_tl = inp_grid.get_corner(UL)   # top-left corner of input grid

        out_cells = [obj for obj in out_grid if isinstance(obj, Square)]
        cell_idx = 0

        patch = None
        for i in range(OUT):
            for j in range(OUT):
                new_patch = highlight_patch(inp_tl, i, j, K, CS)

                if patch is None:
                    self.play(FadeIn(new_patch), run_time=0.25)
                else:
                    self.play(Transform(patch, new_patch), run_time=0.25)
                    patch = new_patch

                if patch is None:
                    patch = new_patch

                # light up the corresponding output cell
                out_cells[cell_idx].set_fill(YCOLOR, opacity=0.55)
                self.play(Flash(out_cells[cell_idx],
                                color=YCOLOR, flash_radius=CS * 0.7,
                                line_length=CS * 0.35, run_time=0.2))
                cell_idx += 1

        self.wait(1)
        self.play(FadeOut(VGroup(inp_group, k_group, out_group,
                                 star1, eq, dot_eq, patch)))

    # -----------------------------------------------------------------------
    # Scene 4 – Layer-2 formula (multi-channel input)
    # -----------------------------------------------------------------------
    def _scene_layer2_formula(self):
        head = Tex(r"\textbf{Layer 2}  —  multi-channel input",
                   color=WHITE).scale(0.85).to_edge(UP, buff=0.35)

        tensors = MathTex(
            r"Y_1 \in \mathbb{R}^{C_1 \times H_1 \times W_1}",
            r"\quad",
            r"K^{(2)} \in \mathbb{R}^{C_2 \times C_1 \times k \times k}",
            color=WHITE).scale(0.65)
        tensors[0].set_color(XCOLOR)
        tensors[2].set_color(KCOLOR)
        tensors.next_to(head, DOWN, buff=0.35)

        formula = MathTex(
            r"Y_2[c,\,i,\,j]",
            r"=",
            r"\sum_{c'=0}^{C_1-1}",
            r"\sum_{m=0}^{k-1}\sum_{n=0}^{k-1}",
            r"Y_1[c',\;i{+}m,\;j{+}n]",
            r"\cdot",
            r"K^{(2)}[c,\,c',\,m,\,n]",
            color=WHITE).scale(0.62)
        formula[0].set_color(YCOLOR)
        formula[4].set_color(XCOLOR)
        formula[6].set_color(KCOLOR)
        formula.next_to(tensors, DOWN, buff=0.45)

        channel_note = MathTex(
            r"\underbrace{\sum_{c'}}_{\text{sum over input channels}}",
            color=GRAY).scale(0.65).next_to(formula, DOWN, buff=0.5)

        dim2 = MathTex(
            r"Y_2 \in \mathbb{R}^{C_2 \times H_2 \times W_2}",
            color=YCOLOR).scale(0.65).next_to(channel_note, DOWN, buff=0.35)

        self.play(Write(head))
        self.play(FadeIn(tensors, shift=UP * 0.15))
        self.play(Write(formula), run_time=2)
        self.play(FadeIn(channel_note))
        self.play(FadeIn(dim2))
        self.wait(2)
        self.play(FadeOut(VGroup(head, tensors, formula,
                                 channel_note, dim2)))

    # -----------------------------------------------------------------------
    # Scene 5 – visual stack summary
    # -----------------------------------------------------------------------
    def _scene_summary(self):
        title = Tex(r"Two-Layer Conv2D Pipeline", color=WHITE).scale(0.78)\
                    .to_edge(UP, buff=0.4)

        def box(text, color):
            lbl = MathTex(text, color=color).scale(0.62)
            rect = SurroundingRectangle(lbl, color=color,
                                        buff=0.22, corner_radius=0.1)
            return VGroup(rect, lbl)

        b_x  = box(r"X\;(1 \times 32 \times 32)",  XCOLOR)
        b_k1 = box(r"*\;K^{(1)}\;(C_1\!\times\!1\!\times\!k\!\times\!k)", KCOLOR)
        b_y1 = box(r"Y_1\;(C_1 \times H_1 \times W_1)",                    YCOLOR)
        b_k2 = box(r"*\;K^{(2)}\;(C_2\!\times\!C_1\!\times\!k\!\times\!k)", KCOLOR)
        b_y2 = box(r"Y_2\;(C_2 \times H_2 \times W_2)",                    YCOLOR)

        pipeline = VGroup(b_x, b_k1, b_y1, b_k2, b_y2)\
                        .arrange(DOWN, buff=0.32).center()

        arrows = VGroup(*[
            Arrow(pipeline[i].get_bottom(),
                  pipeline[i + 1].get_top(),
                  buff=0.05, color=GRAY, stroke_width=2,
                  max_tip_length_to_length_ratio=0.25)
            for i in range(len(pipeline) - 1)
        ])

        self.play(Write(title))
        for mob, arr in zip(pipeline, [None] + list(arrows)):
            if arr:
                self.play(GrowArrow(arr), FadeIn(mob, shift=DOWN * 0.15),
                          run_time=0.55)
            else:
                self.play(FadeIn(mob))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, pipeline, arrows)))
