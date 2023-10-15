"""
an animatio nof the original implementation of the flexible layer
"""


from manim import (
    Arrow,
    DOWN,
    FadeIn,
    FadeOut,
    tempconfig,
    Group,
    Write,
    UR,
    TransformFromCopy,
    GrowArrow,
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
)
from manim.utils.file_ops import open_file as open_media_file
from random import randint
from src.tensorspec.matrices.flat import FlatMatrix3D, FlatMatrix3DImage
from pathlib import Path


class Example(MovingCameraScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def construct(self):
        matrix_image = FlatMatrix3DImage(
            image_path=Path("assets", "private_images", "puppy.png"), dimensions=(3, 32, 32), label="Image"
        )
        matrix_conv = FlatMatrix3D(dimensions=(12, 12, 12), label="Conv")
        matrix_pool = FlatMatrix3D(dimensions=(3, 12, 12), label="MaxPool")

        matrix_image.scale_to_fit_width(2).move_to(ORIGIN).shift(LEFT * 5)
        matrix_pool.scale_to_fit_width(matrix_image.width).next_to(matrix_image, UR, buff=0.1)
        matrix_conv.scale_to_fit_width(matrix_image.width).next_to(matrix_image, DR, buff=0.1)

        self.add(matrix_image, matrix_conv, matrix_pool)

        transforms = [FadeIn(matrix_pool, matrix_conv)]

        self.play(*transforms)
        self.wait(1)

        # -----------------------------
        matrix_pool_expanded = FlatMatrix3D(dimensions=(12, 12, 12), label="MaxPool")
        matrix_pool_expanded.scale_to_fit_width(matrix_image.width).next_to(matrix_pool, RIGHT, buff=1.5)

        arrow_expansion = Arrow(matrix_pool.get_right(), matrix_pool_expanded.get_left(), buff=0.1, color=WHITE)
        arrow_text = Text("Interpolation", color=WHITE).move_to(arrow_expansion.get_center()).shift(UP * 0.3)
        arrow_text.scale_to_fit_width(arrow_expansion.width * 0.8)

        self.add(matrix_pool_expanded, arrow_expansion, arrow_text)

        transforms = [
            TransformFromCopy(matrix_pool, matrix_pool_expanded),
            matrix_conv.animate.align_to(matrix_pool_expanded, LEFT),
            GrowArrow(arrow_expansion),
            Write(arrow_text),
        ]
        self.play(*transforms)
        self.wait(1)
        # -----------------------------

        matrix_threshold = FlatMatrix3D(dimensions=(12, 12, 12), label="Thresholds")
        matrix_threshold.scale_to_fit_width(matrix_image.width).next_to(matrix_pool_expanded, RIGHT, buff=1.5)

        self.add(matrix_threshold)
        self.play(FadeIn(matrix_threshold), self.camera.frame.animate.align_to(matrix_pool_expanded, LEFT).shift(LEFT * 3))
        self.wait(1)
        # -----------------------------

        arrow_subs = Arrow(matrix_pool_expanded.get_right(), matrix_threshold.get_left(), buff=0.1, color=WHITE)
        arrow_subs_text = Text("substract by", color=WHITE).move_to(arrow_subs.get_center()).shift(UP * 0.3)
        arrow_subs_text.scale_to_fit_width(arrow_subs.width * 0.8)

        self.add(arrow_subs, arrow_subs_text)

        matrix_logits = FlatMatrix3D(dimensions=(12, 12, 12), label="Logits")
        matrix_logits.scale_to_fit_width(matrix_image.width).next_to(matrix_threshold, RIGHT, buff=1.5)

        arrow_equal_text = Text("=", color=WHITE).next_to(matrix_logits, LEFT, buff=0.5)
        arrow_equal_text.scale_to_fit_width(arrow_equal_text.width * 0.8)

        self.add(matrix_logits, arrow_equal_text)

        transforms = [
            TransformFromCopy(matrix_threshold, matrix_logits),
            GrowArrow(arrow_subs),
            Write(arrow_subs_text),
            Write(arrow_equal_text),
        ]
        self.play(*transforms)
        self.wait(1)

        # -----------------------------
        matrix_mask = FlatMatrix3D(dimensions=(12, 12, 12), label="Binary Mask")
        matrix_mask.scale_to_fit_width(matrix_image.width).next_to(matrix_logits, RIGHT, buff=1.5)

        arrow_sigmoid = Arrow(matrix_logits.get_right(), matrix_mask.get_left(), buff=0.1, color=WHITE)
        arrow_sigmoid_text = Text("Sigmoid * 50", color=WHITE).move_to(arrow_sigmoid.get_center()).shift(UP * 0.3)
        arrow_sigmoid_text.scale_to_fit_width(arrow_sigmoid.width * 0.8)

        self.add(matrix_mask, arrow_sigmoid, arrow_sigmoid_text)

        transforms = [
            TransformFromCopy(matrix_logits, matrix_mask),
            GrowArrow(arrow_sigmoid),
            Write(arrow_sigmoid_text),
            self.camera.frame.animate.align_to(matrix_logits, LEFT).shift(LEFT * 3),
        ]
        self.play(*transforms)
        self.wait(1)

        # -----------------------------
        transforms = [
            matrix_mask.animate.align_to(matrix_threshold, LEFT).align_to(matrix_image, DOWN),
            self.camera.frame.animate.align_to(matrix_pool_expanded, LEFT).shift(LEFT * 3),
            FadeOut(
                arrow_subs,
                arrow_subs_text,
                arrow_equal_text,
                matrix_logits,
                matrix_threshold,
                arrow_sigmoid,
                arrow_sigmoid_text,
            ),
        ]
        self.play(*transforms)
        self.wait(1)

        # multiply conv --------
        matrix_conv_copy = matrix_conv.copy().next_to(matrix_mask, DR, buff=0.1).shift(RIGHT * 0.3)
        transofrms = [Transform(matrix_conv, matrix_mask)]
        self.play(*transofrms)
        self.wait(1)

        # output -----------------------------
        transforms = [ReplacementTransform(matrix_conv, matrix_conv_copy)]
        self.play(*transforms)
        self.wait(1)

        # reverse the mask
        matrix_mask_reverse = FlatMatrix3D(dimensions=(12, 12, 12), label="1 - Binary Mask")
        matrix_mask_reverse.scale_to_fit_width(matrix_image.width).move_to(matrix_mask.get_center())

        transforms = [ReplacementTransform(matrix_mask, matrix_mask_reverse)]
        self.play(*transforms)
        self.wait(1)

        # multiple pool -------
        matrix_pool_expanded_copy = matrix_pool_expanded.copy().next_to(matrix_mask_reverse, UR, buff=0.1).shift(RIGHT * 0.3)
        transofrms = [Transform(matrix_pool_expanded, matrix_mask_reverse)]
        self.play(*transofrms)
        self.wait(1)

        # output -----------------------------
        transforms = [ReplacementTransform(matrix_pool_expanded, matrix_pool_expanded_copy)]
        self.play(*transforms)
        self.wait(1)

        # sum
        matrix_sum = FlatMatrix3D(dimensions=(12, 12, 12), label="Sum")
        matrix_sum.scale_to_fit_width(matrix_image.width).align_to(matrix_mask_reverse, DOWN).align_to(
            matrix_pool_expanded_copy, LEFT
        ).shift(RIGHT * 2)

        transforms = [
            ReplacementTransform(Group(matrix_pool_expanded_copy, matrix_conv_copy), matrix_sum),
            self.camera.frame.animate.shift(RIGHT * 3),
        ]
        self.play(*transforms)
        self.wait(1)

        # ---- zoom out
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
