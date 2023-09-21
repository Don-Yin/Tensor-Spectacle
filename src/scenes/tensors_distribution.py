from manim import ThreeDScene, ORIGIN, LEFT, RIGHT, ReplacementTransform, UP
from src.components.plots import create_distribution_plot
from src.components.matrices import create_3D_matrix as create_3D_matrix_native
from src.components.matrices_plt import create_3D_matrix as create_3D_matrix_plt
from src.components.labels import make_labels


class TensorVisualizationScene(ThreeDScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_opengl = self.renderer.__class__.__name__ == "OpenGLRenderer"

        if self.use_opengl:
            setattr(self.camera, "frame_width", self.get_camera_width())
            setattr(self.camera, "frame_height", self.get_camera_height())

    def construct(self, tensors: list, labels: list, duration_each: float, duration_gap: float, engine: str = "matplotlib"):
        """
        Visualizes a list of tensors with associated labels using 3D matrix representations and distribution plots.

        This function takes in a list of tensors and their associated labels and visualizes them in 3D space.
        Each tensor is represented as both a 3D matrix and a distribution plot. The rendering can be sped up
        using either Manim's native 3D matrix methods or a faster version using matplotlib.

        Parameters:
        ----------
        tensors : list
            List of tensors to be visualized. Each tensor should be an array-like structure suitable for
            visualization as a 3D matrix.

        labels : list
            List of labels associated with each tensor. Should be the same length as tensors.

        duration_each : float
            Duration (in seconds) for each transition from one tensor representation to the next.

        duration_gap : float
            Duration (in seconds) for the gap between transitions.

        engine : str, optional (default: "matplotlib")
            The backend engine to be used for creating the 3D matrix visualization.
            Available options are:
            - "matplotlib": Uses the matplotlib library for a faster rendering.
            - "native": Uses Manim's native methods for creating the 3D matrix. This option might be slower.

        Raises:
        ------
        AssertionError:
            - If `tensors` is not a list.
            - If the lengths of `tensors` and `labels` are different.
            - If `engine` is neither "matplotlib" nor "native".

        Notes:
        -----
        Using the matplotlib engine can save hours of rendering time compared to the native method.
        This function supports both OpenGL and Cairo renderers, and will adjust camera attributes accordingly.

        Examples:
        --------
        >>> class Example(TensorVisualizationScene):
            def construct(self):
                random_shapes = [(randint(1, 2), randint(3, 12), randint(3, 12), randint(3, 12)) for _ in range(3)]
                tensors = [torch.rand(shape) for shape in random_shapes]
                labels = [f"Random Tensor {i}" for i in range(len(tensors))]
                super().construct(tensors=tensors, labels=labels, duration_each=0.8, duration_gap=1, engine="matplotlib")
        """
        assert isinstance(tensors, list), f"tensors must be a list, got {type(tensors)}"
        assert len(tensors) == len(labels), f"tensors and labels must have the same length, got {len(tensors)} / {len(labels)}"
        assert engine in ["matplotlib", "native"], f"engine must be either 'matplotlib' or 'native', got {engine}"

        match engine:
            case "matplotlib":
                create_3D_matrix = create_3D_matrix_plt
            case "native":
                create_3D_matrix = create_3D_matrix_native
                
        num_tensors = len(tensors)
        barplots = []
        cubes = []

        for tensor in tensors:
            # -------- create the distribution plot --------
            bar_group = (
                create_distribution_plot(tensor, use_opengl_renderer=self.use_opengl)
                .move_to(ORIGIN)
                .shift(RIGHT * self.get_camera_width() * 0.25)
            )
            bar_group = self.scale_to_fit_camera(bar_group, width_ratio=0.36, height_ratio=0.8)
            barplots.append(bar_group)
            

            # -------- create the 3D matrix --------
            cube_group = (
                create_3D_matrix(tensor, use_opengl_renderer=self.use_opengl)
                .move_to(ORIGIN)
                .shift(LEFT * self.get_camera_width() * 0.25)
            )
            cube_group = self.scale_to_fit_camera(cube_group, width_ratio=0.36, height_ratio=0.8)
            cubes.append(cube_group)

        # -------- labels --------
        labels = [make_labels(labels, selected_idx=i).to_edge(UP, buff=0.1) for i in range(num_tensors)]

        # -------- tensors --------
        self.add(barplots[0], cubes[0], labels[0])

        for i in range(num_tensors - 1):
            self.play(
                ReplacementTransform(barplots[i], barplots[i + 1]),
                ReplacementTransform(cubes[i], cubes[i + 1]),
                ReplacementTransform(labels[i], labels[i + 1]),
                run_time=duration_each,
            )
            self.wait(duration_gap)
        self.wait()

    # -------- renderer specific camera setting --------
    def scale_to_fit_camera(self, mobject, width_ratio=0.5, height_ratio=0.5):
        """
        scale mobject to fit the camera frame depending on which side of the mobject is larger: the larger side will shrink.
        """
        if mobject.get_width() > mobject.get_height():
            return mobject.scale_to_fit_width(self.get_camera_width() * width_ratio)
        return mobject.scale_to_fit_height(self.get_camera_height() * height_ratio)

    def get_camera_width(self):
        """
        opengl and cairo have different camera attributes
        """
        if self.use_opengl:
            return self.camera.frame_shape[0]
        return self.camera.frame_width

    def get_camera_height(self):
        if self.use_opengl:
            return self.camera.frame_shape[1]
        return self.camera.frame_height
