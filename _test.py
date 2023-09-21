# from src.tensorspec.scenes.tensors_distribution import TensorVisualizationScene
from src.tensorspec.scenes.tensors_distribution import TensorVisualizationScene
from src.tensorspec.utils.general import banner
from manim.utils.file_ops import open_file as open_media_file
from manim import tempconfig
from random import randint
import torch


class Example(TensorVisualizationScene):
    def construct(self):
        """
        # time taken for matplotlib: 120.86233758926392s
        # time taken for native: 865.6813843250275s
        """
        random_shapes = [(randint(1, 2), randint(3, 6), randint(3, 6), randint(3, 6)) for _ in range(3)]
        tensors = [torch.rand(shape) for shape in random_shapes]
        labels = [f"Random Tensor {i}" for i in range(len(tensors))]
        banner("Starting - Depending on the complexity of the tensors, this may take a while")
        # engine: "native" / matplotlib
        super().construct(tensors=tensors, labels=labels, duration_each=0.8, duration_gap=1, engine="matplotlib")


def test_example_runs():
    with tempconfig(
        {
            "use_opengl_renderer": True,
            "disable_caching": True,
            "write_to_movie": True,
            "resolution": "1080p",
            "renderer": "opengl",
            "fps": 60,
        }
    ):
        scene = Example()
        scene.render()
        open_media_file(scene.renderer.file_writer.movie_file_path)


if __name__ == "__main__":
    test_example_runs()
