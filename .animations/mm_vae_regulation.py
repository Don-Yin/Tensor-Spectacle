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
    Group,
    ORIGIN,
    ImageMobject,
    FadeIn,
    GrowArrow,
)
from manim.utils.file_ops import open_file as open_media_file
from src.tensorspec.node.net import NetNode


import toml
from manim import Arrow, DOWN, BLACK, Arrow, Create, Tex, WHITE, Text, UP, ORIGIN, Transform, RIGHT, LEFT
from random import randint
from src.tensorspec.vae.mm_vae import MM_VAE
from src.tensorspec.matrices.flat import FlatMatrix3DImage
from src.tensorspec.node.net import NetNode


class Example(MovingCameraScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.font_size = 24

    def construct(self):
        self.camera.background_color = WHITE
        # make ====
        vae = MM_VAE(font_size=self.font_size).move_to(ORIGIN)

        flat_fmri = (
            FlatMatrix3DImage(
                image_path=Path("assets", "fmri.jpg"),
                dimensions=(3, 32, 32),
                main_color=BLACK,
                font_color=BLACK,
                label="fMRI",
            )
            .scale_to_fit_height(vae.encoders[0].height)
            .next_to(vae.encoders[0], LEFT * 5)
        )

        flat_mri = (
            FlatMatrix3DImage(
                image_path=Path("assets", "mri.jpeg"),
                dimensions=(3, 32, 32),
                main_color=BLACK,
                font_color=BLACK,
                label="MRI",
            )
            .scale_to_fit_height(vae.encoders[1].height)
            .next_to(vae.encoders[1], LEFT * 5)
        )

        node_genetic_data = NetNode(text="Genetic Data", text_color=BLACK, fill_color=BLACK, text_font_size=27).next_to(
            vae.encoders[2], LEFT * 5
        )

        # make arrows
        arrows_in = [
            Arrow(from_.get_right(), to.get_left(), color=BLACK, stroke_width=2)
            for from_, to in zip([flat_fmri, flat_mri, node_genetic_data], vae.encoders)
        ]

        flat_fmri_copy = (
            FlatMatrix3DImage(
                image_path=Path("assets", "fmri.jpg"),
                dimensions=(3, 32, 32),
                main_color=BLACK,
                font_color=BLACK,
                label="Reconstructed fMRI",
            )
            .scale_to_fit_height(vae.encoders[0].height)
            .next_to(vae.decoders[0], RIGHT * 5)
        )

        flat_mri_copy = (
            FlatMatrix3DImage(
                image_path=Path("assets", "mri.jpeg"),
                dimensions=(3, 32, 32),
                main_color=BLACK,
                font_color=BLACK,
                label="Reconstructed MRI",
            )
            .scale_to_fit_height(vae.encoders[1].height)
            .next_to(vae.decoders[1], RIGHT * 5)
        )

        node_genetic_data_copy = NetNode(
            text="Reconstructed Genetic Data", text_color=BLACK, fill_color=BLACK, text_font_size=27, rec_width=4
        ).next_to(vae.decoders[2], RIGHT * 5)

        arrows_out = [
            Arrow(from_.get_right(), to.get_left(), color=BLACK, stroke_width=2)
            for from_, to in zip(vae.decoders, [flat_fmri_copy, flat_mri_copy, node_genetic_data_copy])
        ]

        # node regulation
        node_regulation = NetNode(
            text="Loss Regulation with Mixture Gaussian Modelling",
            text_color=BLACK,
            fill_color=BLACK,
            text_font_size=27,
            rec_width=7,
        ).next_to(vae.latent_distribution_group, UP * 9)

        arrow_regulation = Arrow(vae.latent_label.get_top(), node_regulation.get_bottom(), color=BLACK, stroke_width=2)
        self.add(arrow_regulation)

        # images
        flat_1 = ImageMobject("media/images/flat_1.png").scale_to_fit_height(2)
        flat_2 = ImageMobject("media/images/flat_2.png").scale_to_fit_height(2)
        img_3d_1 = ImageMobject("media/images/3d_1.png").scale_to_fit_height(2)
        img_3d_2 = ImageMobject("media/images/3d_2.png").scale_to_fit_height(2)

        flat_group = Group(flat_1, flat_2).arrange(RIGHT, buff=2).next_to(node_regulation, UP * 4)
        img_3d_group = Group(img_3d_1, img_3d_2).arrange(RIGHT, buff=2).next_to(flat_group, UP * 2)

        flat_1.shift((img_3d_1.get_center()[0] - flat_1.get_center()[0]) * RIGHT)
        flat_2.shift((img_3d_2.get_center()[0] - flat_2.get_center()[0]) * RIGHT)

        group_1 = Group(flat_1, img_3d_1)
        group_2 = Group(flat_2, img_3d_2)

        arrow_1_2 = Arrow(group_1.get_right(), group_2.get_left(), color=BLACK, stroke_width=2)
        # arrow_regulation_img = Arrow(node_regulation.get_top(), flat_group.get_bottom(), color=BLACK, stroke_width=2)
        # self.add(arrow_regulation_img)

        # animate ----
        transforms = [
            *node_regulation.on_create(),
            GrowArrow(arrow_1_2),
            FadeIn(flat_1),
            FadeIn(flat_2),
            FadeIn(img_3d_1),
            FadeIn(img_3d_2),
            *vae.on_create(),
            *flat_fmri.on_create(),
            *flat_mri.on_create(),
            *flat_fmri_copy.on_create(),
            *flat_mri_copy.on_create(),
            *node_genetic_data_copy.on_create(),
            *node_genetic_data.on_create(),
            self.camera.frame.animate.scale(2).shift(LEFT, 0.2).shift(UP, 1.2),
        ]
        for arrow in arrows_in:
            transforms += [Create(arrow)]

        for arrow in arrows_out:
            transforms += [Create(arrow)]

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
