"""
VAE vs Supervised Learning
"""

from pathlib import Path
from itertools import product
import toml
from manim import tempconfig, WHITE, MovingCameraScene, LEFT
from manim.utils.file_ops import open_file as open_media_file


import toml
from manim import (
    Arrow,
    DOWN,
    BLACK,
    Group,
    Mobject,
    Write,
    Arrow,
    Create,
    Tex,
    WHITE,
    Text,
    UP,
    ORIGIN,
    Transform,
    RIGHT,
    GrowArrow,
)
from random import randint
from src.tensorspec.vae.mm_vae import MM_VAE_WITH_DATA
from src.tensorspec.vae.coders import Coder
from src.tensorspec.relation.arrow import ConnectionArrow
from src.tensorspec.node.net import NetNode


class Example(MovingCameraScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.font_size = 24

    def construct(self):
        self.camera.background_color = WHITE
        # make ====
        mm_vae = MM_VAE_WITH_DATA(font_size=self.font_size).move_to(ORIGIN)
        node_discriminator = Coder(edge_ratio=0.5, rotate_degrees=90, label=f"Discriminator", font_size=19).scale(1.2)
        node_discriminator.next_to(mm_vae.vae.vector, DOWN * 7).shift(LEFT * 2)
        arrow_disc = ConnectionArrow(
            _from=(mm_vae.vae.vector, DOWN), to=(node_discriminator, RIGHT), color=BLACK, stroke_width=3
        )
        self.add(node_discriminator, arrow_disc, mm_vae)

        # add object
        text_objective = Text("From which data source?", color=BLACK, font_size=self.font_size).scale(1.2)
        text_objective.next_to(node_discriminator, LEFT * 5)
        arrow_objective = Arrow(node_discriminator.get_left(), text_objective.get_right(), color=BLACK, stroke_width=3)
        self.add(text_objective, arrow_objective)

        # data sources
        # GENFI / ADNI \citep{ADNI}, UK Biobank \citep{UKBiobank2018}, and CamCAN
        sources = ["GENFI", "ADNI", "UK Biobank", "CamCAN"]
        nodes_sources = [
            NetNode(text=source, text_color=BLACK, fill_color=BLACK, text_font_size=27).scale_to_fit_width(
                mm_vae.vae.encoders[0].width
            )
            for source in sources
        ]
        group_sources = (
            Group(*nodes_sources)
            .arrange(DOWN)
            .next_to(Group(mm_vae.flat_fmri, mm_vae.flat_mri, mm_vae.node_genetic_data), LEFT * 5)
        )
        self.add(group_sources)

        # make arrows for all combinations
        combinations = list(
            product(
                [i.get_right() for i in nodes_sources],
                [i.get_left() for i in [mm_vae.flat_fmri, mm_vae.flat_mri, mm_vae.node_genetic_data]],
            )
        )
        arrows_in = [Arrow(*combo, color=BLACK, max_tip_length_to_length_ratio=0, stroke_width=3) for combo in combinations]
        self.add(*arrows_in)

        # animate ----
        transforms = [
            self.camera.frame.animate.scale(2).shift(LEFT * 2),
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
