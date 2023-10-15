import math
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
    FadeTransformPieces,
    Transform,
    LEFT,
    RIGHT,
    Create,
)
from manim.utils.file_ops import open_file as open_media_file
from src.tensorspec.utils.general import banner
from random import randint
from torch import tensor
import math
from src.tensorspec.matrices.flat import FlatMatrix3D, FlatMatrix3DImage
from pathlib import Path


class Example(MovingCameraScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def construct(self):
        image = FlatMatrix3DImage(image_path=Path("assets", "puppy.png"), dimensions=(3, 32, 32), label="Image")
        image.scale_to_fit_width(self.camera.frame_width * 0.2).move_to(ORIGIN)

        image_2 = FlatMatrix3D(dimensions=(3, 32, 32), label="Image")
        image_2.scale_to_fit_width(self.camera.frame_width * 0.2).move_to(LEFT * 4)


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
