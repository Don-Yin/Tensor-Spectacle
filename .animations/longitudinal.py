#!/Users/donyin/miniconda3/envs/tensor/bin/python

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
    DOWN,
    UL,
    DL,
)
from manim.utils.file_ops import open_file as open_media_file
from src.tensorspec.vae.vae import VAE
from src.tensorspec.node.net import NetNode
from src.tensorspec.relation.bound_box import bounding_moject
from src.tensorspec.relation.arrow import ConnectionArrow


import toml
from manim import Arrow, BLACK, Group, Mobject, Write, Arrow, Create, Tex, WHITE, Text, UP, ORIGIN, Transform, RIGHT, LEFT
from random import randint
from src.tensorspec.vae.mm_vae import MM_VAE, MM_VAE_WITH_DATA
from src.tensorspec.matrices.flat import FlatMatrix3DImage, FlatMatrix3D
from src.tensorspec.node.net import NetNode
from src.tensorspec.relation.repeat import repeat_mobject
from src.tensorspec.vae.coders import Coder


class Example(MovingCameraScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.font_size = 32

    def construct(self):
        self.camera.background_color = WHITE
        # make ====
        mm_vae = MM_VAE_WITH_DATA(font_size=self.font_size).move_to(ORIGIN)
        self.add(mm_vae)

        # add bbox and label
        bbox = bounding_moject(
            Group(mm_vae.data_in_1, mm_vae.data_in_2, mm_vae.data_in_3),
            label="Longitudinal Data",
            font_size=self.font_size,
        )

        self.add(bbox)

        # -------- now get the stacked embeddings ---------
        stacked_embeddings = (
            FlatMatrix3D(
                dimensions=(1, 10, 32),
                main_color=WHITE,
                font_color=BLACK,
                label=r"Dementia Progression Embedding\\One Patient",
                # total_channel_thickness=32,
            )
            .scale_to_fit_height(mm_vae.vae.vector.height)
            .next_to(mm_vae.vae.vector, DOWN * 12)
        )
        stacked_embeddings.text_str.scale(0.7).shift(UP * 0.3)
        self.add(stacked_embeddings)

        # arrow time indicator
        arrow_time = Arrow(stacked_embeddings.get_left(), stacked_embeddings.get_right(), color=BLACK, stroke_width=3)
        arrow_time = arrow_time.next_to(stacked_embeddings, DOWN)
        self.add(arrow_time)

        label_time = Tex(r"temporal dimension", color=BLACK, font_size=self.font_size + 8).next_to(arrow_time, DOWN * 0.3)
        self.add(label_time)

        # arrow from the vector to the stacked embeddings
        arrow_vector_to_stacked = Arrow(
            mm_vae.vae.vector.get_bottom(), stacked_embeddings.text_str.get_top(), color=BLACK, stroke_width=3
        )
        self.add(arrow_vector_to_stacked)

        # ------------ multiple patients situation --------------
        stacked_embeddings_indi = stacked_embeddings.copy().align_to(bbox, LEFT).shift(DOWN * (stacked_embeddings.height + 3))
        self.add(stacked_embeddings_indi)

        # multiple patients
        stacked_embeddings_mul = repeat_mobject(stacked_embeddings_indi.matrix, 16, 0.05).next_to(
            stacked_embeddings_indi, RIGHT * 7
        )
        self.add(stacked_embeddings_mul)
        label_stacked_mul = Tex(
            r"Dementia Progression Embedding\\Multiple Patients", color=BLACK, font_size=self.font_size + 6
        ).next_to(stacked_embeddings_mul, UP)
        self.add(label_stacked_mul)

        # clustering node
        node_clustering = NetNode(
            text=r"Unsupervised\\Clustering\\(e.g., SuStaIn)",
            text_color=BLACK,
            fill_color=BLACK,
            text_font_size=self.font_size - 3,
            rec_width=4,
        ).next_to(stacked_embeddings_mul, RIGHT * 5)

        # arrows from stacked embeddings to clustering
        arrow_stacked_to_clustering = Arrow(
            stacked_embeddings_mul.get_right(), node_clustering.get_left(), color=BLACK, stroke_width=3
        )
        arrow_ind_to_mul = Arrow(
            stacked_embeddings_indi.get_right(), stacked_embeddings_mul.get_left(), color=BLACK, stroke_width=3
        )

        # text_stack = Tex(r"Stacked", color=BLACK, font_size=self.font_size).next_to(arrow_ind_to_mul, UP * 0.2)

        self.add(node_clustering, arrow_stacked_to_clustering, arrow_ind_to_mul)

        # group and align
        group_section_init = Group(
            mm_vae,
            mm_vae.data_in_1,
            mm_vae.data_in_2,
            mm_vae.data_in_3,
            bbox,
            stacked_embeddings,
            arrow_time,
            label_time,
            arrow_vector_to_stacked,
        )

        group_section_cluster = (
            Group(
                stacked_embeddings_indi,
                stacked_embeddings_mul,
                label_stacked_mul,
                node_clustering,
                arrow_stacked_to_clustering,
                arrow_ind_to_mul,
            )
            .scale_to_fit_width(group_section_init.width)
            .align_to(group_section_init, LEFT)
        )

        # discriminator
        node_discriminator = Coder(
            edge_ratio=0.5, rotate_degrees=90, label=f"Discriminator", font_size=self.font_size - 8
        ).scale(1.2)
        node_discriminator.next_to(mm_vae.vae.vector_label, UP * 6).shift(LEFT * 2)
        arrow_disc = ConnectionArrow(
            _from=(mm_vae.vae.vector_label, UP), to=(node_discriminator, RIGHT), color=BLACK, stroke_width=3
        )
        self.add(node_discriminator, arrow_disc)

        text_objective = Tex(r"From which data dataset\\(e.g., CamCAN)?", color=BLACK, font_size=self.font_size).scale(1.2)
        text_objective.next_to(node_discriminator, LEFT * 5)
        arrow_objective = Arrow(node_discriminator.get_left(), text_objective.get_right(), color=BLACK, stroke_width=3)
        self.add(text_objective, arrow_objective)

        # animate ----
        transforms = [
            self.camera.frame.animate.scale(2.5).shift(DOWN * 2.4),
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
