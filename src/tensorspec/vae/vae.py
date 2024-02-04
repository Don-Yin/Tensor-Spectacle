from manim import Arrow, DOWN, BLACK, Group, Mobject, Write, Arrow, Create, Tex, WHITE, Text, UP, ORIGIN, Transform, RIGHT, LEFT
from src.tensorspec.matrices.flat import FlatMatrix3D, FlatMatrix3DImage
from src.tensorspec.vae.coders import Coder
from src.tensorspec.node.net import NetNode
import inspect
from pathlib import Path


class VAE(Mobject):
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
        # make ====
        self.encoder = Coder(edge_ratio=0.5, rotate_degrees=-90, label="Encoder").move_to(ORIGIN)
        self.latent_mu = NetNode(text=r"$\mu$", text_color=BLACK, fill_color=BLACK, text_font_size=42)
        self.latent_sigma = NetNode(text=r"$\sigma$", text_color=BLACK, fill_color=BLACK, text_font_size=42)
        self.latent_sigma = self.latent_sigma.next_to(self.latent_mu, DOWN * 4)
        self.latent_distribution = Group(self.latent_mu, self.latent_sigma).scale(0.7).next_to(self.encoder, RIGHT)
        self.latent_label = Text("Latent Distribution", color=BLACK, font_size=self.font_size)
        self.latent_label = self.latent_label.next_to(self.latent_distribution, UP)

        self.add(self.encoder, self.latent_mu, self.latent_sigma, self.latent_label)

        # make ====
        self.vector = FlatMatrix3D(dimensions=(1, 10, 1), label="", main_color=WHITE).scale_to_fit_height(
            self.latent_distribution.get_height()
        )
        self.vector = self.vector.next_to(self.latent_distribution, RIGHT * 8)
        self.vector_label = Tex("Vector", color=BLACK, font_size=self.font_size).next_to(self.vector, UP)
        self.vector_label.align_to(self.latent_label, DOWN)

        self.add(self.vector, self.vector_label)

        # make ====
        self.node_sample = Arrow(self.latent_distribution.get_right(), self.vector.get_left(), color=BLACK, stroke_width=2)
        self.node_sample_label = Tex(r"Sampling $\mathbf{z}$", color=BLACK, font_size=self.font_size)
        self.node_sample_label = self.node_sample_label.next_to(self.node_sample, UP)

        self.add(self.node_sample, self.node_sample_label)

        # make ====
        self.decoder = Coder(edge_ratio=0.5, rotate_degrees=90, label="Decoder").next_to(self.vector, RIGHT)
        self.decoder.align_to(self.encoder, UP)

        self.add(self.decoder)

    def on_create(self):
        transforms = [
            *self.encoder.on_create(),
            *self.latent_mu.on_create(),
            *self.latent_sigma.on_create(),
            Write(self.latent_label),
            *self.vector.on_create(),
            Write(self.vector_label),
            Create(self.node_sample),
            Write(self.node_sample_label),
            *self.decoder.on_create(),
        ]

        return transforms
