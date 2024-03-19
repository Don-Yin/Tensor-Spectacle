"""
VAE vs Supervised Learning
"""
from pathlib import Path
from itertools import product
import toml
from manim import (
    tempconfig,
    Tex,
    WHITE,
    MovingCameraScene,
    Text,
    ORIGIN,
)
from manim.utils.file_ops import open_file as open_media_file
from src.tensorspec.vae.vae import VAE
from src.tensorspec.node.net import NetNode


import toml
from manim import Arrow, BLACK, Group, Mobject, Write, Arrow, Create, Tex, WHITE, Text, UP, ORIGIN, Transform, RIGHT, LEFT
from random import randint
from src.tensorspec.vae.mm_vae import MM_VAE, MM_VAE_WITH_DATA
from src.tensorspec.matrices.flat import FlatMatrix3DImage
from src.tensorspec.node.net import NetNode


class Example(MovingCameraScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.font_size = 24

    def construct(self):
        self.camera.background_color = WHITE
        # make ====
        mm_vae = MM_VAE_WITH_DATA(font_size=self.font_size).move_to(ORIGIN)

        self.add(mm_vae)

        # animate ----
        transforms = [
            self.camera.frame.animate.scale(1.8),
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
