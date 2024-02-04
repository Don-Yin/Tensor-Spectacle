"""
Encoder and decoder mojects for VAEs.
https://docs.manim.community/en/stable/reference/manim.mobject.geometry.polygram.Polygon.html?highlight=polygon
"""

import inspect
from manim import (
    UP,
    DOWN,
    LEFT,
    RIGHT,
    DEGREES,
    Arrow,
    Mobject,
    VMobject,
    LineJointType,
    FadeIn,
    Polygon,
    WHITE,
    Create,
    BLACK,
    Text,
    Write,
)


class Coder(Mobject):
    """
    Encoder or Decoder for VAEs.
    """

    def __init__(self, *args, **kwargs):
        # ================== edge ratio / maybe orientation later ===============
        self.edge_ratio = kwargs.get("edge_ratio", 0.5)
        self.rotate_degrees = kwargs.get("rotate_degrees", 0)
        self.label = kwargs.get("label", "")
        self.font_size = kwargs.get("font_size", 24)
        # ================== pop kwargs for parent ==================
        parent_params = inspect.signature(super().__init__).parameters
        [kwargs.pop(kw) for kw in list(kwargs.keys()) if kw not in parent_params]
        super().__init__(*args, **kwargs)
        # ===========================================================
        self._make_trapezoid()

    def _make_trapezoid(self):
        bottom_length = 2
        top_length = bottom_length * self.edge_ratio
        height = 2
        bottom_left = [-bottom_length / 2, 0, 0]
        bottom_right = [bottom_length / 2, 0, 0]
        top_left = [-top_length / 2, height, 0]
        top_right = [top_length / 2, height, 0]
        self.trapezoid = Polygon(bottom_left, bottom_right, top_right, top_left, color=BLACK, fill_opacity=0.5)
        self.trapezoid.rotate(DEGREES * self.rotate_degrees)

        self.label_text = Text(self.label, color=BLACK, font_size=self.font_size)
        self.label_text.move_to(self.trapezoid.get_center())

        self.add(self.trapezoid, self.label_text)

    def on_create(self):
        transforms = [Create(self.trapezoid), Write(self.label_text)]
        return transforms
