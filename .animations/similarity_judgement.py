# mindset here
# https://docs.manim.community/en/stable/reference/manim.mobject.geometry.polygram.RoundedRectangle.html
import math
import numpy as np
from manim import (
    Arrow,
    BLACK,
    DoubleArrow,
    Matrix,
    MathTex,
    DOWN,
    FadeOut,
    tempconfig,
    Unwrite,
    Group,
    Write,
    ArrowCircleTip,
    GrowArrow,
    ReplacementTransform,
    WHITE,
    MovingCameraScene,
    Text,
    UP,
    ORIGIN,
    LEFT,
    RIGHT,
    Create,
    BLUE,
)
from manim.utils.file_ops import open_file as open_media_file
from random import randint
import math
from src.tensorspec.matrices.flat import FlatMatrix3D, FlatMatrix3DImage
from pathlib import Path
from src.tensorspec.node.net import NetNode


class Decoder(MovingCameraScene):
    def construct(self):
        # ================== title ==================
        title_text = "Measuring Similarity Judgements in Neural Networks"
        title_node = Text(title_text).move_to(ORIGIN).scale_to_fit_width(self.camera.frame_width * 0.7)

        self.play(Write(title_node), run_time=1)
        self.wait(1)
        self.play(Unwrite(title_node), run_time=1)

        # ================== Merging Block 1 ==================
        example_name_strs = [
            "3x3 conv, 64",
            "3x3 conv, 64",
            "max pool",
        ]

        example_node_list = [NetNode(text=i).scale_to_fit_width(self.camera.frame_width * 0.3) for i in example_name_strs]
        example_group = Group(*example_node_list).arrange(DOWN, buff=0.3).move_to(ORIGIN)

        for node in example_node_list:
            self.play(*node.on_create(), self.camera.frame.animate.move_to(node), run_time=0.5)
            self.wait(0.2)

        self.play(
            self.camera.frame.animate.move_to(example_group.get_bottom()).shift(UP * example_group.height * 0.5),
            run_time=1,
        )

        # ================== merge first ==================
        blocks_strs = ["Block 1", "Block 2", "Block 3", "Block 4", "Block 5", "fc 4096", "fc 4096", "fc 1000"]
        blocks_nodes = [NetNode(text=i).scale_to_fit_width(self.camera.frame_width * 0.4) for i in blocks_strs]
        blocks_group = Group(*blocks_nodes).arrange(DOWN, buff=0.3).move_to(ORIGIN)

        self.play(
            self.camera.frame.animate.move_to(blocks_nodes[0]),
            ReplacementTransform(example_group, blocks_nodes[0]),
            run_time=1,
        )

        # ================== make all blocks ==================
        for i in range(1, len(blocks_nodes)):
            self.play(
                *blocks_nodes[i].on_create(),
                self.camera.frame.animate.move_to(blocks_nodes[i]),
                run_time=0.3,
            )
            self.wait(0.2)

        # ================== zoom ==================
        self.play(
            self.camera.frame.animate.move_to(blocks_group.get_bottom()).shift(UP * blocks_group.height * 0.5),
            blocks_group.animate.scale_to_fit_height(self.camera.frame.height * 0.8),
            run_time=1,
        )

        # ================== text introduction ==================
        text_intro = "VGG16"
        text_node = Text(text_intro, color=WHITE).scale_to_fit_width(self.camera.frame_width * 0.2)
        text_node.next_to(blocks_group, RIGHT, buff=3)
        text_blocks_group = Group(blocks_group, text_node)

        text_any_other = "(or any other networks)"
        text_any_other_node = Text(text_any_other, color=WHITE).scale_to_fit_width(self.camera.frame_width * 0.13)
        text_any_other_node.next_to(text_node, DOWN, buff=0.2)

        self.play(
            self.camera.frame.animate.move_to(text_blocks_group.get_center()),
            Write(text_node),
            Write(text_any_other_node),
            run_time=1,
        )

        self.wait(1)

        # ================== zoom out ==================
        self.play(
            Unwrite(text_node),
            Unwrite(text_any_other_node),
            run_time=1,
        )

        # ================== rotate to horizontal ==================
        self.play(
            blocks_group.animate.rotate(math.pi / 2),
            self.camera.frame.animate.move_to(blocks_group.get_center()),
            run_time=1,
        )

        # ================== add the image ==================
        input_image_1 = (
            FlatMatrix3DImage(image_path=Path("assets", "L.png"), dimensions=(3, 32, 32), label="Image 1")
            .scale_to_fit_height(blocks_nodes[0].height * 2)
            .next_to(blocks_nodes[0], LEFT, buff=1.3)
        )
        self.play(
            *input_image_1.on_create(),
            self.camera.frame.animate.shift(LEFT * 3),
            run_time=1,
        )

        # ================== add the arrow and rolling colors ==================
        progress_arrow = Arrow(start=input_image_1.get_right(), end=blocks_nodes[0].get_left(), buff=0.1).scale(0.7)
        self.play(GrowArrow(progress_arrow), run_time=1)
        self.wait(0.3)

        self.play(
            progress_arrow.animate.rotate(-math.pi / 2).next_to(blocks_nodes[0], UP, buff=0.1),
            run_time=0.3,
        )

        self.play(
            FadeOut(progress_arrow),
            run_time=0.3,
        )

        for block in blocks_nodes:
            self.play(
                block.animate._set_fill(BLUE, opacity=0.3),
                self.camera.frame.animate.move_to(block),
                run_time=0.1,
            )
            self.play(block.animate._set_fill(BLACK, opacity=0), run_time=0.2)

        # ================== getting the output ==================
        prediction = (
            Matrix([[np.random.rand().__round__(3)], ["..."], [np.random.rand().__round__(3)], [np.random.rand().__round__(3)]])
            .next_to(blocks_nodes[-1], RIGHT, buff=2)
            .scale_to_fit_height(blocks_nodes[0].height * 0.8)
        )

        result_arrow = Arrow(start=blocks_nodes[-1].get_right(), end=prediction.get_left(), buff=0.1).scale(0.7)

        self.play(
            # *prediction.on_create(),
            Create(prediction),
            GrowArrow(result_arrow),
            run_time=1,
        )
        self.wait(1)
        # ================== now we start content specific creations ==================
        image_and_blocks_nodes = [input_image_1, *blocks_nodes[:-1]]
        image_and_blocks_group = Group(*image_and_blocks_nodes)  # exclude the fc layer
        fc_node = blocks_nodes[-1]

        self.play(
            image_and_blocks_group.animate.shift(LEFT * 2),
            run_time=1,
        )

        # ================== fc connection ==================
        block_fc_connect_arrow = DoubleArrow(
            start=image_and_blocks_nodes[-1].get_right(),
            end=fc_node.get_left(),
            buff=0.1,
            tip_shape_start=ArrowCircleTip,
            tip_shape_end=ArrowCircleTip,
            max_tip_length_to_length_ratio=0.1,
        ).scale(0.7)
        self.play(
            GrowArrow(block_fc_connect_arrow),
            self.camera.frame.animate.move_to(block_fc_connect_arrow),
            run_time=1,
        )

        # ================== take out the intermediate logits ==================
        logits_node_1 = (
            FlatMatrix3D(dimensions=(64, 1, 1), label="Image 1\nPenultimate\nOutput")
            .scale_to_fit_width(block_fc_connect_arrow.width * 0.8)
            ._hide_dimensions()
        )
        logits_node_1.next_to(block_fc_connect_arrow, UP, buff=1)._resize_label(self.camera.frame_width * 0.07)

        logits_node_2 = (
            FlatMatrix3D(dimensions=(64, 1, 1), label="Image 2\nPenultimate\nOutput")
            .scale_to_fit_width(block_fc_connect_arrow.width * 0.8)
            ._hide_dimensions()
        )
        logits_node_2.move_to(logits_node_1)._resize_label(self.camera.frame_width * 0.07)

        logits_out_arrow = DoubleArrow(
            start=block_fc_connect_arrow.get_top(), end=logits_node_1.get_bottom(), buff=0.1, tip_shape_start=ArrowCircleTip
        )

        self.play(
            *logits_node_1.on_create(),
            GrowArrow(logits_out_arrow),
            self.camera.frame.animate.move_to(block_fc_connect_arrow),
            run_time=1,
        )

        self.wait(1)

        # ================== collect the output ==================
        self.play(
            logits_node_1.animate.shift(UP + LEFT),
            self.camera.frame.animate.shift(UP + LEFT),
            run_time=1,
        )
        self.wait(1)

        # ================== show image 2 ==================
        input_image_2 = (
            FlatMatrix3DImage(image_path=Path("assets", "L1.png"), dimensions=(3, 32, 32), label="Image 2")
            .scale_to_fit_height(input_image_1.height)
            .next_to(input_image_1, DOWN, buff=0.5)
        )
        self.play(
            self.camera.frame.animate.move_to(input_image_2),
            run_time=1,
        )
        self.play(
            *input_image_2.on_create(),
            run_time=1,
        )

        # ================== move ==================
        self.play(
            input_image_2.animate.move_to(input_image_1),
            input_image_1.animate.move_to(input_image_2),
            self.camera.frame.animate.move_to(blocks_nodes[0].get_left()),
            run_time=1,
        )

        # ================== process 2 ==================
        progress_arrow = Arrow(start=input_image_2.get_right(), end=blocks_nodes[0].get_left(), buff=0.1).scale(0.7)
        self.play(GrowArrow(progress_arrow), run_time=1)
        self.wait(0.3)
        self.play(FadeOut(progress_arrow), run_time=0.3)

        for block in blocks_nodes[:-1] + [block_fc_connect_arrow, logits_out_arrow]:
            try:
                self.play(
                    block.animate._set_fill(BLUE, opacity=0.3),
                    self.camera.frame.animate.move_to(block),
                    run_time=0.2,
                )
                self.play(block.animate._set_fill(BLACK, opacity=0), run_time=0.1)
            except AttributeError:
                self.play(
                    block.animate.set_fill(BLUE, opacity=0.3),
                    self.camera.frame.animate.move_to(block),
                    run_time=0.2,
                )
                self.play(block.animate.set_fill(WHITE, opacity=1), run_time=0.1)

        self.play(
            *logits_node_2.on_create(),
            run_time=1,
        )

        self.play(
            logits_node_2.animate.next_to(logits_node_1, RIGHT, buff=0.5),
            self.camera.frame.animate.move_to(logits_node_1).shift(RIGHT * 0.5),
            run_time=1,
        )

        # ======== m ========
        matrix_1 = [
            [np.random.rand().__round__(3)],
            ["..."],
            [np.random.rand().__round__(3)],
            [np.random.rand().__round__(3)],
            [np.random.rand().__round__(3)],
        ]
        matrix_2 = [
            [np.random.rand().__round__(3)],
            ["..."],
            [np.random.rand().__round__(3)],
            [np.random.rand().__round__(3)],
            [np.random.rand().__round__(3)],
        ]

        logits_matrix_1 = Matrix(matrix_1).move_to(logits_node_1).scale_to_fit_width(self.camera.frame_width * 0.07)
        logits_matrix_2 = Matrix(matrix_2).move_to(logits_node_2).scale_to_fit_width(self.camera.frame_width * 0.07)
        self.play(
            ReplacementTransform(logits_node_1, logits_matrix_1),
            ReplacementTransform(logits_node_2, logits_matrix_2),
            run_time=1,
        )

        cosine_text = Text(r"Euclidean Distance")
        cosine_text.next_to(logits_matrix_1, RIGHT, buff=3).scale_to_fit_width(self.camera.frame_width * 0.3)
        arrow_cosine = Arrow(start=logits_matrix_2.get_right(), end=cosine_text.get_left(), buff=0.1).scale(0.7)

        self.play(
            Write(cosine_text),
            GrowArrow(arrow_cosine),
            self.camera.frame.animate.move_to(cosine_text),
            run_time=1,
        )
        self.wait(0.5)

        cosine_fomula = MathTex("d(a, b) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}")
        cosine_fomula.move_to(cosine_text).scale_to_fit_width(cosine_text.width)

        self.play(
            ReplacementTransform(cosine_text, cosine_fomula),
            run_time=1,
        )
        self.wait(0.5)

        result = "= .49"
        result_node = MathTex(result).next_to(cosine_fomula, RIGHT, buff=0.3).scale_to_fit_width(self.camera.frame_width * 0.1)
        self.play(
            Write(result_node),
            run_time=1,
        )
        self.wait(1)

        # ================== explain whats going on ==================
        explain_text = "This process is applied at each layer"
        explain_text_node = Text(explain_text).scale_to_fit_width(self.camera.frame_width * 0.6)
        explain_text_node.next_to(cosine_text, UP, buff=1.5)

        self.play(
            Write(explain_text_node),
            self.camera.frame.animate.move_to(explain_text_node),
            run_time=2,
        )

        self.wait(5)


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
