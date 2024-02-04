"""
Script making the stack of 3D cubes at the tensor -> 3D matrix section using manim native cubes.
"""

from manim import VGroup, Text, DOWN, DEGREES, Cube, BLUE, RED, UP, color_to_rgb, rgb_to_color, Mobject
from torch import tensor
import numpy as np


def interpolate_color(color1, color2, alpha):
    """Interpolate between two colors based on an alpha value."""
    rgb1 = np.array(color_to_rgb(color1))
    rgb2 = np.array(color_to_rgb(color2))
    interpolated_rgb = (1 - alpha) * rgb1 + alpha * rgb2
    return rgb_to_color(interpolated_rgb)


class Matrix3DNative(Mobject):
    def __init__(self, tensor: tensor, use_opengl_renderer: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert tensor.dim() == 4, f"The tensor must be 4-dimensional (batch, c, h, w) got {tensor.dim()}"
        self.cube_side_length = 0.1
        self.spacing = 0.12
        self.tensor = tensor
        global VGroup, VMobject
        if use_opengl_renderer:
            from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup as VGroup, OpenGLVMobject as VMobject
        self.make_matrix()

    def make_matrix(self):
        batch_groups = VGroup()
        for batch_idx in range(tensor.shape[0]):
            cube_batch_group = VGroup()
            for i in range(tensor.shape[1]):
                for j in range(tensor.shape[2]):
                    for k in range(tensor.shape[3]):
                        value = tensor[batch_idx, i, j, k].item()
                        cube_color = interpolate_color(BLUE, RED, value)
                        cube = Cube(
                            side_length=self.cube_side_length,
                            fill_opacity=0.5,
                            fill_color=cube_color,
                        )
                        cube.shift(np.array([i * self.spacing, j * self.spacing, k * self.spacing]))
                        cube_batch_group.add(cube)

            # Adding shape label below each cube batch
            shape_label = (
                Text(f"{tensor.shape[1:]}", font_size=8)
                .rotate(30 * DEGREES, axis=[0, 1, 0])
                .set_width(cube_batch_group.get_width())
                .next_to(cube_batch_group, direction=DOWN, buff=0.1)
            )

            # Create a group containing both the cube batch and the label
            batch_group_with_label = VGroup(cube_batch_group, shape_label)
            # The space + x is a buffer between batches, adjust as needed
            batch_group_with_label.shift(UP * batch_idx * (tensor.shape[1] * self.spacing + 0.2))
            batch_groups.add(batch_group_with_label)
        batch_groups.rotate(-30 * DEGREES, axis=[0, 1, 0])
        self.add(batch_groups)
