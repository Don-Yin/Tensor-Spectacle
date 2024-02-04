import inspect
import numpy as np
from manim import RoundedRectangle, Text, VMobject, WHITE, Write, Create, MarkupText, Tex


class NetNode(VMobject):
    """
    This is a mobject shaped like a rectangle with rounded corners with text in the middle.
    """

    def __init__(self, *args, **kwargs):
        self.text = kwargs.get("text", "Node")
        self.text_color = kwargs.get("text_color", WHITE)
        self.node_color = kwargs.get("node_color", WHITE)
        self.text_font_size = kwargs.get("text_font_size", 24)
        self.rec_height = kwargs.get("rec_height", 1)
        self.rec_width = kwargs.get("rec_width", 2)
        # ================== pop kwargs for parent ==================
        parent_params = inspect.signature(super().__init__).parameters
        [kwargs.pop(kw) for kw in list(kwargs.keys()) if kw not in parent_params]
        # pop and feed to super; mobject doesn't take dimensions as an arg; only in base
        super().__init__(*args, **kwargs)
        # ===========================================================
        self.make_node()

    def make_node(self):
        self.node_rec = RoundedRectangle(
            corner_radius=0.1,
            height=self.rec_height,
            width=self.rec_width,
            stroke_width=2,
            fill_color=self.node_color,
            stroke_color=self.text_color,
        )
        self.text_obj = Tex(self.text, color=self.text_color, font_size=self.text_font_size).move_to(self.node_rec.get_center())
        self.add(self.node_rec, self.text_obj)

    # def the behaviour under create
    def on_create(self):
        return [Write(self.text_obj), Create(self.node_rec)]

    def _set_fill(self, *args, **kwargs):
        self.node_rec.set_fill(*args, **kwargs)
        return self
