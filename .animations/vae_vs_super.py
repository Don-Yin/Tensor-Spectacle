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
    GrowArrow,
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
        # vae = VAE(font_size=self.font_size).move_to(ORIGIN)
        vae = NetNode(text="Model", text_color=BLACK, fill_color=BLACK, text_font_size=27).move_to(ORIGIN)

        # animate ----
        transforms = [*vae.on_create()]
        self.play(*transforms)
        self.wait(1)

        # make =====
        node_input_data = NetNode(text="Input", text_color=BLACK, fill_color=BLACK, text_font_size=27)
        node_input_data.next_to(vae, LEFT * 5)

        # quesiton
        text_question = Text("Which subtype?", color=BLACK, font_size=self.font_size).next_to(vae, RIGHT * 5)
        arrow_question = Arrow(vae.get_right(), text_question.get_left(), color=BLACK, stroke_width=2)

        node_subtypes = [NetNode(text=i, text_color=BLACK, fill_color=BLACK, text_font_size=27) for i in ["AD", "MCI", "NC"]]
        node_subtypes_group = Group(*node_subtypes).arrange(DOWN, buff=0.4).next_to(text_question, RIGHT * 5)

        combinations = list(product([i.get_left() for i in node_subtypes], [text_question.get_right()]))
        self.subtype_arrows = [
            Arrow(*combo, color=BLACK, max_tip_length_to_length_ratio=0, stroke_width=3) for combo in combinations
        ]
        self.add(*self.subtype_arrows)

        arrow_input = Arrow(node_input_data.get_right(), vae.get_left(), color=BLACK, stroke_width=2)

        # animate ----
        transforms = [
            *node_input_data.on_create(),
            Create(arrow_input),
            GrowArrow(arrow_question),
            Write(text_question),
            self.camera.frame.animate.move_to(arrow_question.get_center()).scale(1.3),
        ]
        for i in node_subtypes:
            transforms += [*i.on_create()]
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
