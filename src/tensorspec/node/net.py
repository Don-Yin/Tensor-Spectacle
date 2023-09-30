import inspect
import numpy as np
from manim import RoundedRectangle, Text, VMobject, WHITE, Write, Create


class NetNode(VMobject):
    """
    This is a mobject shaped like a rectangle with rounded corners with text in the middle.
    """

    def __init__(self, *args, **kwargs):
        self.text = kwargs.get("text")
        # ================== pop kwargs for parent ==================
        parent_params = inspect.signature(super().__init__).parameters
        [kwargs.pop(kw) for kw in list(kwargs.keys()) if kw not in parent_params]
        # pop and feed to super; mobject doesn't take dimensions as an arg; only in base
        super().__init__(*args, **kwargs)
        # ===========================================================
        self.make_node()
        np.random.seed(0)

    def make_node(self):
        self.node_rec = RoundedRectangle(corner_radius=0.1, height=2, width=5, stroke_width=1, fill_color=self.fill_color)
        self.text_obj = Text(self.text, color=WHITE).move_to(self.node_rec.get_center())
        self.text_obj.scale_to_fit_width(self.node_rec.width * 0.06 * len(self.text))

        self.add(self.node_rec, self.text_obj)

    # def the behaviour under create
    def on_create(self):
        return [Write(self.text_obj), Create(self.node_rec)]

    def _set_fill(self, *args, **kwargs):
        self.node_rec.set_fill(*args, **kwargs)
        return self
