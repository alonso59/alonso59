"""
CNN Architecture Visualizer  —  Manim Community v0.20+
Replicates the "3-D feature-map volume" style seen in CNN explainer videos,
showing each layer as a 3-D box whose dimensions reflect the tensor shape
(C × H × W), with shape labels, operation labels, and transitions.

Architecture (LeNet-style on 32×32 MNIST input):
  Input      1  × 32 × 32
  Conv1      6  × 28 × 28   (6 filters, 5×5, valid)
  MaxPool1   6  × 14 × 14   (2×2)
  Conv2      16 × 10 × 10   (16 filters, 5×5, valid)
  MaxPool2   16 ×  5 ×  5   (2×2)
  Flatten    400
  FC1        120
  FC2         84
  Output      10
"""

from manim import *
import numpy as np

# ── palette ──────────────────────────────────────────────────────────────────
BG       = BLACK
C_INPUT  = "#4FC3F7"   # light-blue
C_CONV   = "#FFD54F"   # amber
C_POOL   = "#EF9A9A"   # red-ish
C_FC     = "#A5D6A7"   # green
C_OUT    = "#CE93D8"   # purple
C_ARROW  = "#90A4AE"   # grey
C_LABEL  = WHITE
C_SHAPE  = "#B0BEC5"   # dim grey


# ── helpers ───────────────────────────────────────────────────────────────────

def scale_dim(v, ref=32, max_size=2.4, min_size=0.18):
    """Map a spatial dimension to a visual length (log scale)."""
    return max(min_size, max_size * np.log2(v + 1) / np.log2(ref + 1))


def scale_channels(c, ref=16, max_size=1.8, min_size=0.12):
    """Map channel count to a visual depth."""
    return max(min_size, max_size * np.log2(c + 1) / np.log2(ref + 1))


def feature_map_box(channels, height, width, color, opacity=0.75):
    """
    Return a VGroup that looks like a 3-D rectangular prism in isometric style.
    The box is centred at ORIGIN.
      - width  → horizontal (X)
      - height → vertical   (Y)
      - depth  → channels   (Z, rendered as a diagonal offset)
    """
    w = scale_dim(width)
    h = scale_dim(height)
    d = scale_channels(channels)

    # isometric offset direction
    offset = (RIGHT * 0.5 + UP * 0.3) * d

    # three faces
    def face(corners, fc, stroke=WHITE, sw=1.2):
        p = Polygon(*corners, fill_color=fc, fill_opacity=opacity,
                    stroke_color=stroke, stroke_width=sw)
        return p

    # front face (W × H)
    bl = np.array([-w / 2, -h / 2, 0])
    br = np.array([ w / 2, -h / 2, 0])
    tr = np.array([ w / 2,  h / 2, 0])
    tl = np.array([-w / 2,  h / 2, 0])

    front = face([bl, br, tr, tl], color)

    # top face (W × D)
    mc = ManimColor(color)
    top = face([tl, tr, tr + offset, tl + offset],
               interpolate_color(mc, ManimColor(WHITE), 0.25))

    # right face (D × H)
    right = face([br, br + offset, tr + offset, tr],
                 interpolate_color(mc, ManimColor(BLACK), 0.25))

    grp = VGroup(right, top, front)
    grp.center()
    return grp, w, h, d, offset


# ── per-layer spec ────────────────────────────────────────────────────────────

LAYERS = [
    dict(name="Input",    op="",               C=1,   H=32, W=32, color=C_INPUT),
    dict(name="Conv1",    op=r"*K^{(1)}_{6\times1\times5\times5}",
                                                C=6,   H=28, W=28, color=C_CONV),
    dict(name="MaxPool1", op=r"\text{MaxPool}_{2\times2}",
                                                C=6,   H=14, W=14, color=C_POOL),
    dict(name="Conv2",    op=r"*K^{(2)}_{16\times6\times5\times5}",
                                                C=16,  H=10, W=10, color=C_CONV),
    dict(name="MaxPool2", op=r"\text{MaxPool}_{2\times2}",
                                                C=16,  H=5,  W=5,  color=C_POOL),
]

FC_LAYERS = [
    dict(name="Flatten", neurons=400, color=C_FC),
    dict(name="FC1",     neurons=120, color=C_FC),
    dict(name="FC2",     neurons=84,  color=C_FC),
    dict(name="Output",  neurons=10,  color=C_OUT),
]


# ── main scene ────────────────────────────────────────────────────────────────

class CNNArchitecture(Scene):

    def construct(self):
        self.camera.background_color = BG
        self._title_card()
        self._conv_layers()
        self._fc_layers()
        self._full_pipeline()

    # ── 0  title ─────────────────────────────────────────────────────────────
    def _title_card(self):
        t = Text("CNN on MNIST  32×32", color=WHITE, font_size=42)
        sub = MathTex(r"X \in \mathbb{R}^{1\times32\times32}",
                      color=C_INPUT, font_size=32).next_to(t, DOWN, buff=0.3)
        arch = MathTex(
            r"\text{Conv}\to\text{Pool}\to\text{Conv}\to\text{Pool}"
            r"\to\text{Flatten}\to\text{FC}^3",
            color=C_SHAPE, font_size=26).next_to(sub, DOWN, buff=0.3)
        self.play(Write(t), run_time=1.0)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.play(FadeIn(arch, shift=UP * 0.2))
        self.wait(1.5)
        self.play(FadeOut(VGroup(t, sub, arch)))

    # ── 1  conv + pool layers one-by-one ─────────────────────────────────────
    def _conv_layers(self):
        for i, layer in enumerate(LAYERS):
            self._show_single_layer(layer, i)

    def _show_single_layer(self, layer, idx):
        C, H, W = layer["C"], layer["H"], layer["W"]
        color    = layer["color"]
        name     = layer["name"]

        box, w, h, d, offset = feature_map_box(C, H, W, color)
        box.move_to(ORIGIN + LEFT * 0.6)

        # ── name top-left
        title = Text(name, color=color, font_size=34).to_edge(UP, buff=0.4)

        # ── shape label  C × H × W
        shape_tex = MathTex(
            rf"{C}", r"\times", rf"{H}", r"\times", rf"{W}",
            color=WHITE, font_size=36)
        shape_tex[0].set_color(C_CONV if C > 1 else C_INPUT)
        shape_tex.next_to(box, RIGHT, buff=0.55)

        # dimension arrows + text
        w_arrow = DoubleArrow(
            box.get_corner(DL), box.get_corner(DR),
            color=C_SHAPE, stroke_width=1.5, tip_length=0.12,
            buff=0.05).shift(DOWN * 0.15)
        w_lbl = MathTex(rf"W={W}", color=C_SHAPE, font_size=20)\
                        .next_to(w_arrow, DOWN, buff=0.08)

        h_arrow = DoubleArrow(
            box.get_corner(DL), box.get_corner(UL),
            color=C_SHAPE, stroke_width=1.5, tip_length=0.12,
            buff=0.05).shift(LEFT * 0.15)
        h_lbl = MathTex(rf"H={H}", color=C_SHAPE, font_size=20)\
                        .next_to(h_arrow, LEFT, buff=0.08)

        # channel brace (along depth offset direction)
        front_tl = box.get_corner(UL)
        back_tl  = front_tl + offset
        c_line   = Line(front_tl, back_tl, color=C_SHAPE, stroke_width=1.5)
        c_lbl    = MathTex(rf"C={C}", color=C_SHAPE, font_size=20)\
                           .next_to(back_tl, UP, buff=0.1)

        # operation label (what produced this layer)
        op_grp = VGroup()
        if layer["op"]:
            op_tex = MathTex(layer["op"], color=C_ARROW, font_size=24)\
                             .to_edge(DOWN, buff=0.45)
            op_grp.add(op_tex)

        self.play(Write(title), run_time=0.5)
        self.play(FadeIn(box), run_time=0.6)
        self.play(
            GrowArrow(w_arrow), Write(w_lbl),
            GrowArrow(h_arrow), Write(h_lbl),
            Create(c_line),     Write(c_lbl),
            run_time=0.7)
        self.play(Write(shape_tex), run_time=0.6)
        if op_grp:
            self.play(FadeIn(op_grp, shift=UP * 0.1))
        self.wait(1.2)

        self.play(FadeOut(VGroup(
            title, box, shape_tex,
            w_arrow, w_lbl, h_arrow, h_lbl,
            c_line, c_lbl, op_grp)))

    # ── 2  FC layers ──────────────────────────────────────────────────────────
    def _fc_layers(self):
        for fc in FC_LAYERS:
            self._show_fc_layer(fc)

    def _show_fc_layer(self, fc):
        n     = fc["neurons"]
        color = fc["color"]
        name  = fc["name"]

        MAX_DOTS = 12
        n_show   = min(n, MAX_DOTS)
        spacing  = 0.38
        dots = VGroup(*[
            Circle(radius=0.13, fill_color=color,
                   fill_opacity=0.85, stroke_width=0)
            for _ in range(n_show)
        ]).arrange(DOWN, buff=0.08).move_to(ORIGIN)

        title = Text(name, color=color, font_size=34).to_edge(UP, buff=0.4)

        shape_tex = MathTex(
            rf"\mathbb{{R}}^{{{n}}}", color=WHITE, font_size=36)\
            .next_to(dots, RIGHT, buff=0.55)

        if n > MAX_DOTS:
            ellipsis = MathTex(r"\vdots", color=C_SHAPE,
                               font_size=28).next_to(dots, DOWN, buff=0.05)
        else:
            ellipsis = VGroup()

        self.play(Write(title), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(d, shift=RIGHT * 0.1)
                                for d in dots], lag_ratio=0.05), run_time=0.7)
        if ellipsis:
            self.play(FadeIn(ellipsis))
        self.play(Write(shape_tex), run_time=0.5)
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, dots, shape_tex, ellipsis)))

    # ── 3  full pipeline panorama ─────────────────────────────────────────────
    def _full_pipeline(self):
        title = Text("Full Pipeline", color=WHITE, font_size=36)\
                     .to_edge(UP, buff=0.35)
        self.play(Write(title))

        # Build all boxes scaled to fit in one scene
        all_layers = [
            ("Input",    C_INPUT, 1,  32, 32),
            ("Conv1",    C_CONV,  6,  28, 28),
            ("Pool1",    C_POOL,  6,  14, 14),
            ("Conv2",    C_CONV,  16, 10, 10),
            ("Pool2",    C_POOL,  16,  5,  5),
        ]

        # scale everything down to fit
        SCALE = 0.45
        boxes   = []
        labels  = []
        shapes  = []

        x_pos = -5.8
        prev_right = None
        arrows = VGroup()

        for name, col, C, H, W in all_layers:
            box, w, h, d, off = feature_map_box(C, H, W, col, opacity=0.8)
            box.scale(SCALE).move_to([x_pos + (w * SCALE) / 2, -0.2, 0])

            lbl = Text(name, color=col, font_size=16)\
                       .next_to(box, UP, buff=0.12)
            shp = MathTex(
                rf"{C}\times{H}\times{W}",
                color=C_SHAPE, font_size=14)\
                .next_to(box, DOWN, buff=0.12)

            boxes.append(box)
            labels.append(lbl)
            shapes.append(shp)

            if prev_right is not None:
                arr = Arrow(prev_right, box.get_left(),
                            buff=0.06, color=C_ARROW,
                            stroke_width=1.5,
                            max_tip_length_to_length_ratio=0.3)
                arrows.add(arr)

            prev_right = box.get_right()
            x_pos += w * SCALE + d * SCALE * 0.5 + 0.55

        # FC blocks as thin vertical bars
        fc_specs = [
            ("Flatten\n400", C_FC,  400),
            ("FC1\n120",     C_FC,  120),
            ("FC2\n84",      C_FC,   84),
            ("Out\n10",      C_OUT,  10),
        ]
        fc_boxes  = []
        fc_labels = []

        for fc_name, fc_col, n in fc_specs:
            bar_h = max(0.18, 2.0 * np.log2(n + 1) / np.log2(401)) * SCALE * 1.8
            bar = Rectangle(width=0.22, height=bar_h,
                            fill_color=fc_col, fill_opacity=0.85,
                            stroke_width=0.8, stroke_color=WHITE)
            bar.move_to([x_pos + 0.11, -0.2, 0])
            fc_lbl = Text(fc_name, color=fc_col, font_size=13)\
                          .next_to(bar, DOWN, buff=0.1)
            arr = Arrow(prev_right, bar.get_left(),
                        buff=0.06, color=C_ARROW,
                        stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.3)
            arrows.add(arr)
            fc_boxes.append(bar)
            fc_labels.append(fc_lbl)
            prev_right = bar.get_right()
            x_pos += 0.65

        all_mobs = VGroup(
            *boxes, *labels, *shapes,
            *fc_boxes, *fc_labels, arrows)

        # centre in frame
        all_mobs.center().shift(DOWN * 0.15)

        self.play(
            LaggedStart(
                *[FadeIn(b) for b in boxes],
                lag_ratio=0.15),
            run_time=1.5)
        self.play(
            LaggedStart(
                *[FadeIn(VGroup(l, s)) for l, s in zip(labels, shapes)],
                lag_ratio=0.15),
            run_time=1.0)
        self.play(Create(arrows), run_time=1.0)
        self.play(
            LaggedStart(
                *[FadeIn(VGroup(b, l)) for b, l in zip(fc_boxes, fc_labels)],
                lag_ratio=0.2),
            run_time=1.0)
        self.wait(3.0)
        self.play(FadeOut(VGroup(title, all_mobs)))
