"""
an animatio nof the original implementation of the flexible layer
"""

# import line joint type
from manim import *
from manim import (
    Arrow,
    DOWN,
    FadeIn,
    tempconfig,
    Write,
    UR,
    TransformFromCopy,
    VMobject,
    GrowArrow,
    Group,
    ReplacementTransform,
    WHITE,
    DR,
    MovingCameraScene,
    Text,
    UP,
    ORIGIN,
    Transform,
    LEFT,
    RIGHT,
    Mobject,
)
from manim.utils.file_ops import open_file as open_media_file
from random import randint
from src.tensorspec.matrices.flat import FlatMatrix3D, FlatMatrix3DImage
from pathlib import Path
from src.tensorspec.relation.arrow import ConnectionArrow
from src.tensorspec.node.net import NetNode


class Example(MovingCameraScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def construct(self):
        # -------- make the init matrices --------
        matrix_image = FlatMatrix3DImage(
            image_path=Path("assets", "private_images", "puppy.png"), dimensions=(3, 32, 32), label="Image"
        )
        matrix_conv = FlatMatrix3D(dimensions=(12, 12, 12), label="Conv")
        matrix_pool = FlatMatrix3D(dimensions=(3, 12, 12), label="MaxPool")

        matrix_image.scale_to_fit_width(2).move_to(ORIGIN).shift(LEFT * 5)
        matrix_pool.scale_to_fit_width(matrix_image.width).next_to(matrix_image, UR, buff=0.1)
        matrix_conv.scale_to_fit_width(matrix_image.width).next_to(matrix_image, DR, buff=0.1)

        arrow_to_conv = ConnectionArrow(_from=(matrix_image, DOWN), to=(matrix_conv, LEFT), buff=0.1, stroke_width=3)
        arrow_to_pool = ConnectionArrow(_from=(matrix_image, UP), to=(matrix_pool, LEFT), buff=0.1, stroke_width=3)

        transforms = [FadeIn(matrix_pool, matrix_conv), *arrow_to_conv.on_create(), *arrow_to_pool.on_create()]

        self.add(matrix_image)  # since this is stationary, we have to add it manually
        self.play(*transforms)
        self.wait(1)

        # -------- interpolate the pool channel --------
        matrix_pool_2 = FlatMatrix3D(dimensions=(12, 12, 12), label="MaxPool")
        matrix_pool_2.scale_to_fit_width(matrix_image.width).next_to(matrix_pool, RIGHT, buff=1.5)
        matrix_conv_2 = matrix_conv.copy().align_to(matrix_pool_2, LEFT)

        arrow_expansion = Arrow(matrix_pool.get_right(), matrix_pool_2.get_left(), buff=0.1, color=WHITE)
        arrow_text = Text("Interpolation", color=WHITE).move_to(arrow_expansion.get_center()).shift(UP * 0.3)
        arrow_text.scale_to_fit_width(arrow_expansion.width * 0.8)

        arrow_to_conv_new = ConnectionArrow(_from=(matrix_image, DOWN), to=(matrix_conv_2, LEFT), buff=0.1, stroke_width=3)

        self.add(matrix_pool_2, arrow_expansion, arrow_text)

        transforms = [
            ReplacementTransform(arrow_to_conv, arrow_to_conv_new),
            TransformFromCopy(matrix_pool, matrix_pool_2),
            ReplacementTransform(matrix_conv, matrix_conv_2),
            GrowArrow(arrow_expansion),
            Write(arrow_text),
        ]
        self.play(*transforms)
        self.wait(1)
        # -------- introduce the logits component --------
        logits_fun_node = NetNode(text="Logits Function").next_to(Group(matrix_conv_2, matrix_pool_2), RIGHT, buff=0.1)
        logits_fun_node.scale_to_fit_width(matrix_image.width)
        arrow_pool_to_logits = ConnectionArrow(
            _from=(matrix_pool_2, DOWN), to=(logits_fun_node, LEFT), buff=0.1, stroke_width=3
        )
        arrow_conv_to_logits = ConnectionArrow(_from=(matrix_conv_2, UP), to=(logits_fun_node, LEFT), buff=0.1, stroke_width=3)

        self.play(
            self.camera.frame.animate.move_to(logits_fun_node.get_center()),
            *logits_fun_node.on_create(),
            *arrow_pool_to_logits.on_create(),
            *arrow_conv_to_logits.on_create(),
            run_time=1,
        )
        self.wait(1)
        # -------- introduce make the logits matrix --------
        logits_matrix = FlatMatrix3D(dimensions=(12, 12, 12), label="Logits")
        logits_matrix.scale_to_fit_width(matrix_image.width).next_to(logits_fun_node, RIGHT, buff=1)
        arrow_fun_logits = Arrow(logits_fun_node.get_right(), logits_matrix.get_left(), buff=0.1, color=WHITE)

        self.play(
            self.camera.frame.animate.move_to(logits_matrix.get_center()),
            *logits_matrix.on_create(),
            GrowArrow(arrow_fun_logits),
            run_time=1,
        )
        self.wait(1)

        # -------- introduce the masking function --------
        masking_fun_node = NetNode(text="Masking Function")
        masking_fun_node.scale_to_fit_width(matrix_image.width).next_to(logits_matrix, RIGHT, buff=1)
        arrow_logits_to_masking = Arrow(logits_matrix.get_right(), masking_fun_node.get_left(), buff=0.1, color=WHITE)

        self.play(
            self.camera.frame.animate.move_to(masking_fun_node.get_center()),
            *masking_fun_node.on_create(),
            GrowArrow(arrow_logits_to_masking),
            run_time=1,
        )
        self.wait(1)

        # -------- make the binary mask --------
        mask_matrix = FlatMatrix3D(dimensions=(12, 12, 12), label="Binary Mask")
        mask_matrix.scale_to_fit_width(matrix_image.width).next_to(masking_fun_node, RIGHT, buff=1)
        arrow_masking_to_mask = Arrow(masking_fun_node.get_right(), mask_matrix.get_left(), buff=0.1, color=WHITE)

        self.play(
            self.camera.frame.animate.move_to(mask_matrix.get_center()),
            *mask_matrix.on_create(),
            GrowArrow(arrow_masking_to_mask),
            run_time=1,
        )
        self.wait(1)

        # -------- inverse the mask using 1 - mask matrix --------
        mask_inverse = FlatMatrix3D(dimensions=(12, 12, 12), label="Binary Mask Inverse")
        mask_inverse.scale_to_fit_width(matrix_image.width).next_to(mask_matrix, UP, buff=1)
        mask_inverse.align_to(matrix_conv_2, DOWN)
        mask_matrix_2 = mask_matrix.copy().align_to(matrix_pool_2, DOWN)
        arrow_inverse = Arrow(mask_matrix_2.get_bottom(), mask_inverse.get_top(), buff=0.1, color=WHITE)
        arrow_annotation = Text("1 - m", color=WHITE, font_size=20)
        arrow_annotation.next_to(arrow_inverse, RIGHT, buff=0.1)

        self.play(
            self.camera.frame.animate.move_to(arrow_inverse),
            ReplacementTransform(mask_matrix, mask_matrix_2),
            *mask_inverse.on_create(),
            GrowArrow(arrow_inverse),
            Write(arrow_annotation),
            run_time=1,
        )
        self.wait(1)

        # -------- multiply the mask with the masks --------

        # 1. move camera to the pool and conv
        self.play(
            self.camera.frame.animate.align_to(matrix_pool_2, LEFT).shift(LEFT * self.camera.frame.width / 2),
            run_time=1,
        )
        self.wait(1)
        # 2. draw the arrows to the matrices
        arrow_conv_mul = Arrow(matrix_conv_2.get_right(), mask_inverse.get_left(), buff=0.1, color=WHITE)
        arrow_pool_mul = Arrow(matrix_pool_2.get_right(), mask_matrix_2.get_left(), buff=0.1, color=WHITE)
        arrow_annot_conv = Text("×", color=WHITE, font_size=20).next_to(arrow_conv_mul, DOWN, buff=0.1)
        arrow_annot_pool = arrow_annot_conv.copy().next_to(arrow_pool_mul, UP, buff=0.1)

        self.play(
            self.camera.frame.animate.move_to(arrow_inverse),
            GrowArrow(arrow_conv_mul),
            GrowArrow(arrow_pool_mul),
            Write(arrow_annot_conv),
            Write(arrow_annot_pool),
            run_time=1,
        )
        self.wait(1)

        # -------- make the conv and pool after multiplication
        matrix_conv_3 = matrix_conv_2.copy().next_to(mask_inverse, RIGHT, buff=1)
        matrix_pool_3 = matrix_pool_2.copy().next_to(mask_matrix_2, RIGHT, buff=1)
        arrow_apply_mask_conv = Arrow(mask_inverse.get_right(), matrix_conv_3.get_left(), buff=0.1, color=WHITE)
        arrow_apply_mask_pool = Arrow(mask_matrix_2.get_right(), matrix_pool_3.get_left(), buff=0.1, color=WHITE)

        self.play(
            self.camera.frame.animate.move_to(Group(matrix_conv_3, matrix_pool_3)),
            *matrix_pool_3.on_create(),
            *matrix_conv_3.on_create(),
            GrowArrow(arrow_apply_mask_conv),
            GrowArrow(arrow_apply_mask_pool),
            run_time=1,
        )
        self.wait(1)

        # -------- sumup --------
        matrix_sum = FlatMatrix3D(dimensions=(12, 12, 12), label="Sum").scale_to_fit_width(matrix_image.width)
        matrix_sum.next_to(Group(matrix_conv_3, matrix_pool_3), RIGHT, buff=1)
        arrow_sum_conv = ConnectionArrow(
            _from=(matrix_conv_3, RIGHT),
            to=(matrix_sum, DOWN),
            buff=0.1,
            stroke_width=3,
        )
        arrow_sum_pool = ConnectionArrow(
            _from=(matrix_pool_3, RIGHT),
            to=(matrix_sum, UP),
            buff=0.1,
            stroke_width=3,
        )
        self.play(
            self.camera.frame.animate.move_to(matrix_sum),
            *matrix_sum.on_create(),
            *arrow_sum_conv.on_create(),
            *arrow_sum_pool.on_create(),
            run_time=1,
        )
        self.wait(1)

        # -------- save image --------
        group_edges = Group(matrix_image, matrix_sum)
        self.play(
            self.camera.frame.animate.scale_to_fit_width(group_edges.width * 1.1).move_to(group_edges),
            run_time=1,
        )
        self.wait(1)


def test_example_runs():
    with tempconfig(
        {
            "use_opengl_renderer": True,
            "disable_caching": False,
            "write_to_movie": True,
            "pixel_width": 2560,
            "pixel_height": 1440,
            "renderer": "cairo",  # "opengl", "cairo"
            # "save_last_frame": True,  # Add this line; this line actually causes bugs sometimes during rendering
            "fps": 60,
        }
    ):
        scene = Example()
        scene.render()
        open_media_file(scene.renderer.file_writer.movie_file_path)


if __name__ == "__main__":
    test_example_runs()
