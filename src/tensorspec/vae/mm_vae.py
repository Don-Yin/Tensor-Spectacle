"""
VAE vs Supervised Learning
"""

from itertools import product
from manim import (
    Arrow,
    DOWN,
    BLACK,
    Group,
    Mobject,
    Write,
    Arrow,
    GrowArrow,
    Tex,
    WHITE,
    LEFT,
    Text,
    UP,
    ORIGIN,
    Transform,
    RIGHT,
)
from src.tensorspec.node.net import NetNode
from pathlib import Path

import toml
from manim import Arrow, DOWN, BLACK, Group, Mobject, Write, Arrow, Create, Tex, WHITE, Text, UP, ORIGIN, Transform, RIGHT
from src.tensorspec.matrices.flat import FlatMatrix3D, FlatMatrix3DImage
from src.tensorspec.vae.coders import Coder
from src.tensorspec.node.net import NetNode
import inspect


class MM_VAE(Mobject):
    def __init__(self, *args, **kwargs):
        self.font_size = kwargs.get("font_size", 24)
        # ================== pop kwargs for parent ==================
        parent_params = inspect.signature(super().__init__).parameters
        [kwargs.pop(kw) for kw in list(kwargs.keys()) if kw not in parent_params]
        # pop and feed to super; mobject doesn't take dimensions as an arg; only in base
        super().__init__(*args, **kwargs)
        # ===========================================================
        self._make_components()

    def _make_components(self):
        # encoders
        self.encoders = [
            Coder(edge_ratio=0.5, rotate_degrees=-90, label=f"Encoder {i}", font_size=self.font_size).move_to(ORIGIN)
            for i in range(3)
        ]
        self.encoder_group = Group(*self.encoders).arrange(DOWN, buff=0.4)
        self.add(self.encoder_group)

        # first latent
        self.latent_layers = [NetNode(text="Latent", text_color=BLACK, fill_color=BLACK, text_font_size=32) for i in range(3)]
        for i, layer in enumerate(self.latent_layers):
            layer.next_to(self.encoders[i], RIGHT)
        self.latent_layers_group = Group(*self.latent_layers)
        self.add(*self.latent_layers)

        # # make ====
        self.latent_mu = NetNode(text=r"$\mu$", text_color=BLACK, fill_color=BLACK, text_font_size=42)
        self.latent_sigma = NetNode(text=r"$\sigma$", text_color=BLACK, fill_color=BLACK, text_font_size=42)
        self.latent_sigma = self.latent_sigma.next_to(self.latent_mu, DOWN * 4)
        self.latent_distribution_group = Group(self.latent_mu, self.latent_sigma)
        self.latent_distribution_group.scale(0.7).next_to(self.latent_layers_group, RIGHT * 9)
        self.latent_label = Tex(r"Shared Latent Distribution", color=BLACK, font_size=self.font_size - 3)
        self.latent_label = self.latent_label.next_to(self.latent_distribution_group, UP * 2)

        self.add(self.latent_mu, self.latent_sigma, self.latent_label)

        # now connect each element in [latent layer] and [latent distribution] pairwise with arrows
        combinations = list(
            product([i.get_right() for i in self.latent_layers], [self.latent_mu.get_left(), self.latent_sigma.get_left()])
        )
        self.latent_arrows = [
            Arrow(*combo, color=BLACK, max_tip_length_to_length_ratio=0, stroke_width=3, buff=0.4) for combo in combinations
        ]
        self.add(*self.latent_arrows)

        # make ====
        self.vector = FlatMatrix3D(dimensions=(1, 10, 1), label="", main_color=WHITE).scale_to_fit_height(
            self.latent_distribution_group.get_height()
        )
        self.vector = self.vector.next_to(self.latent_distribution_group, RIGHT * 8)
        self.vector_label = Text("Vector", color=BLACK, font_size=self.font_size).next_to(self.vector, UP)

        self.add(self.vector, self.vector_label)

        # make ====
        self.node_sample = Arrow(
            self.latent_distribution_group.get_right(), self.vector.get_left(), color=BLACK, stroke_width=2
        )
        self.node_sample_label = Tex(r"Sampling $\mathbf{z}$", color=BLACK, font_size=self.font_size)
        self.node_sample_label = self.node_sample_label.next_to(self.node_sample, UP)

        self.add(self.node_sample, self.node_sample_label)

        # make ====
        self.decoders = [
            Coder(edge_ratio=0.5, rotate_degrees=90, label=f"Decoder {i}", font_size=self.font_size).next_to(self.vector, RIGHT)
            for i in range(3)
        ]
        self.decoder_group = Group(*self.decoders).arrange(DOWN, buff=0.4).next_to(self.vector, RIGHT * 6)
        for i, decoder in enumerate(self.decoders):
            decoder.align_to(self.encoders[i], DOWN)
        self.add(self.decoder_group)

        # connect vector to each decoder
        self.decoder_arrows = [
            Arrow(
                self.vector.get_right(),
                decoder.get_left(),
                color=BLACK,
                max_tip_length_to_length_ratio=0,
                stroke_width=3,
            )
            for decoder in self.decoders
        ]
        self.add(*self.decoder_arrows)

    def on_create(self):
        transforms = []
        for encoder in self.encoders:
            transforms += [*encoder.on_create()]
        for layer in self.latent_layers:
            transforms += [*layer.on_create()]
        transforms += [*self.latent_mu.on_create()]
        transforms += [*self.latent_sigma.on_create()]
        transforms += [Write(self.latent_label)]
        transforms += [*self.vector.on_create()]
        transforms += [Write(self.vector_label)]
        transforms += [GrowArrow(self.node_sample)]
        transforms += [Write(self.node_sample_label)]
        for decoder in self.decoders:
            transforms += [*decoder.on_create()]
        for arrow in self.latent_arrows:
            transforms += [GrowArrow(arrow)]
        for arrow in self.decoder_arrows:
            transforms += [GrowArrow(arrow)]
        return transforms


class MM_VAE_WITH_DATA(Mobject):
    def __init__(self, *args, **kwargs):
        self.font_size = kwargs.get("font_size", 24)
        self.total_channel_thickness = kwargs.get("total_channel_thickness", 1.5)
        # ================== pop kwargs for parent ==================
        parent_params = inspect.signature(super().__init__).parameters
        [kwargs.pop(kw) for kw in list(kwargs.keys()) if kw not in parent_params]
        super().__init__(*args, **kwargs)
        # -----------------------------------------------------------
        self._make_components()

    def _make_components(self):
        self.vae = MM_VAE(font_size=self.font_size)

        self.data_in_1 = (
            FlatMatrix3DImage(
                image_path=Path("assets", "fmri.jpg"),
                dimensions=(32, 32, 32),
                main_color=BLACK,
                font_color=BLACK,
                label="fMRI",
                total_channel_thickness=32,
            )
            .scale_to_fit_height(self.vae.encoders[0].height)
            .next_to(self.vae.encoders[0], LEFT * 5)
        )

        self.data_in_2 = (
            FlatMatrix3DImage(
                image_path=Path("assets", "mri.jpeg"),
                dimensions=(32, 32, 32),
                main_color=BLACK,
                font_color=BLACK,
                label="Structural MRI",
                total_channel_thickness=32,
            )
            .scale_to_fit_height(self.vae.encoders[1].height)
            .next_to(self.vae.encoders[1], LEFT * 5)
        )

        self.data_in_3 = (
            FlatMatrix3DImage(
                image_path=Path("assets", "rsfMRI.png"),
                dimensions=(32, 32, 32),
                main_color=BLACK,
                font_color=BLACK,
                label="rs-fMRI",
                total_channel_thickness=32,
            )
            .scale_to_fit_height(self.vae.encoders[2].height)
            .next_to(self.vae.encoders[2], LEFT * 5)
        )

        self.arrows_in = [
            Arrow(from_.get_right(), to.get_left(), color=BLACK, stroke_width=2)
            for from_, to in zip([self.data_in_1, self.data_in_2, self.data_in_3], self.vae.encoders)
        ]

        # -------- output --------

        self.data_out_1 = (
            FlatMatrix3DImage(
                image_path=Path("assets", "fmri.jpg"),
                dimensions=(32, 32, 32),
                main_color=BLACK,
                font_color=BLACK,
                label="Reconstructed fMRI",
                total_channel_thickness=32,
            )
            .scale_to_fit_height(self.vae.encoders[0].height)
            .next_to(self.vae.decoders[0], RIGHT * 5)
        )

        self.data_out_2 = (
            FlatMatrix3DImage(
                image_path=Path("assets", "mri.jpeg"),
                dimensions=(32, 32, 32),
                main_color=BLACK,
                font_color=BLACK,
                label="Reconstructed Structural MRI",
                total_channel_thickness=32,
            )
            .scale_to_fit_height(self.vae.encoders[1].height)
            .next_to(self.vae.decoders[1], RIGHT * 5)
        )

        self.data_out_3 = (
            FlatMatrix3DImage(
                image_path=Path("assets", "rsfMRI.png"),
                dimensions=(32, 32, 32),
                main_color=BLACK,
                font_color=BLACK,
                label="Reconstructed rs-fMRI",
                total_channel_thickness=32,
            )
            .scale_to_fit_height(self.vae.encoders[2].height)
            .next_to(self.vae.decoders[2], RIGHT * 5)
        )

        self.arrows_out = [
            Arrow(from_.get_right(), to.get_left(), color=BLACK, stroke_width=2)
            for from_, to in zip(self.vae.decoders, [self.data_out_1, self.data_out_2, self.data_out_3])
        ]

        self.add(
            self.vae,
            *self.arrows_in,
            self.data_in_1,
            self.data_in_2,
            self.data_in_3,
            *self.arrows_out,
            self.data_out_1,
            self.data_out_2,
            self.data_out_3,
        )
