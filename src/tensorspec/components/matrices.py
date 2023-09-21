from manim import VGroup, Text, DOWN, DEGREES, Cube, BLUE, RED, UP, color_to_rgb, rgb_to_color
import numpy as np


def interpolate_color(color1, color2, alpha):
    """Interpolate between two colors based on an alpha value."""
    rgb1 = np.array(color_to_rgb(color1))
    rgb2 = np.array(color_to_rgb(color2))
    interpolated_rgb = (1 - alpha) * rgb1 + alpha * rgb2
    return rgb_to_color(interpolated_rgb)


def create_3D_matrix(tensor, cube_side_length=0.1, spacing=0.12, use_opengl_renderer=False):
    """
    Take a 3D tensor and create a 3D matrix of cubes with the same shape as the tensor.
    Align the matrices in the batch from top to bottom.
    """
    global VGroup, VMobject
    if use_opengl_renderer:
        from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup as VGroup, OpenGLVMobject as VMobject

    assert tensor.dim() == 4, f"The tensor must be 4-dimensional with the first dimension being the batch, got {tensor.dim()}"
    batch_groups = VGroup()
    for batch_idx in range(tensor.shape[0]):
        cube_batch_group = VGroup()
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[2]):
                for k in range(tensor.shape[3]):
                    value = tensor[batch_idx, i, j, k].item()
                    cube_color = interpolate_color(BLUE, RED, value)
                    cube = Cube(
                        side_length=cube_side_length,
                        fill_opacity=0.5,
                        fill_color=cube_color,
                    )
                    cube.shift(np.array([i * spacing, j * spacing, k * spacing]))
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
        batch_group_with_label.shift(UP * batch_idx * (tensor.shape[1] * spacing + 0.2))
        batch_groups.add(batch_group_with_label)

    return batch_groups.rotate(-30 * DEGREES, axis=[0, 1, 0])
