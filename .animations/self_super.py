"""
VAE vs Supervised Learning
"""

import toml
from itertools import product
from manim import (
    Arrow,
    DOWN,
    FadeIn,
    FadeOut,
    tempconfig,
    BLACK,
    Group,
    Mobject,
    Write,
    Arrow,
    Create,
    UR,
    TransformFromCopy,
    GrowArrow,
    ReplacementTransform,
    Tex,
    WHITE,
    DR,
    MovingCameraScene,
    Text,
    UP,
    ORIGIN,
    Transform,
    LEFT,
    RIGHT,
)
from manim.utils.file_ops import open_file as open_media_file
from src.tensorspec.vae.vae import VAE
from src.tensorspec.node.net import NetNode


class Example(MovingCameraScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.font_size = 24

    def construct(self):
        self.camera.background_color = WHITE
        # make ====
        vae = VAE(font_size=self.font_size).move_to(ORIGIN)

        # animate ----
        transforms = [*vae.on_create()]
        self.play(*transforms)
        self.wait(1)

        # make =====
        node_input_data = NetNode(text="Input", text_color=BLACK, fill_color=BLACK, text_font_size=27)
        node_input_data.next_to(vae, LEFT * 5)
        node_reconstruction = NetNode(text="Reconstruction", text_color=BLACK, fill_color=BLACK, text_font_size=27)
        node_reconstruction.next_to(vae, RIGHT * 5)

        arrow_input = Arrow(node_input_data.get_right(), vae.get_left(), color=BLACK, stroke_width=2)
        arrow_reconstruction = Arrow(vae.get_right(), node_reconstruction.get_left(), color=BLACK, stroke_width=4)

        # make model
        model = NetNode(text="Model", text_color=BLACK, fill_color=BLACK, text_font_size=27).move_to(ORIGIN)
        model = model.next_to(vae.vector, DOWN * 5)
        model_arrow = Arrow(vae.vector.get_bottom(), model.get_top(), color=BLACK, stroke_width=2)

        text_question = Text("Which subtype?", color=BLACK, font_size=self.font_size).next_to(model, RIGHT * 5)
        arrow_question = Arrow(model.get_right(), text_question.get_left(), color=BLACK, stroke_width=2)

        node_downstream = NetNode(text="Downstream Tasks", text_color=BLACK, fill_color=BLACK, text_font_size=24, rec_width=3)
        node_downstream.next_to(vae.vector_label, UP * 5)
        arrow_downstream = Arrow(vae.vector_label.get_top(), node_downstream.get_bottom(), color=BLACK, stroke_width=2)

        # animate ----
        transforms = [
            *node_downstream.on_create(),
            GrowArrow(arrow_downstream),
            *node_input_data.on_create(),
            *node_reconstruction.on_create(),
            self.camera.frame.animate.scale(1.3),
            Create(arrow_input),
            Create(arrow_reconstruction),
            *model.on_create(),
            GrowArrow(model_arrow),
            GrowArrow(arrow_question),
            Write(text_question),
        ]
        self.play(*transforms)
        self.wait(1)


def test_example_runs():
    with open("CONFIG.toml", "r") as toml_file:
        config_data = toml.load(toml_file)

    with tempconfig(config_data):
        scene = Example()
        scene.render()
        open_media_file(scene.renderer.file_writer.movie_file_path)


if __name__ == "__main__":
    test_example_runs()
