"""
Script for making the bar plot showing the distribution of tensor values at the tensor -> distribution section
"""


from manim import BarChart, DOWN, Group, Text, UP, VMobject, RIGHT, LEFT
import numpy as np
from scipy.stats import gaussian_kde


def create_distribution_plot(tensor, num_bins=100, font_size=24, width=6, height=4, bw_method=0.1, use_opengl_renderer=False):
    global VGroup, VMobject
    if use_opengl_renderer:
        from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup as VGroup, OpenGLVMobject as VMobject

    assert tensor.dim() == 4, f"The tensor must be 4-dimensional with the first dimension being the batch, got {tensor.dim()}"
    num_batches = tensor.size(0)
    group_list = []

    for i in range(num_batches):
        values = tensor[i].flatten().numpy()
        hist_values, bins = np.histogram(values, bins=np.linspace(0, 1, num_bins + 1))

        # Create the bar chart with different colors for each batch
        barchart = BarChart(values=hist_values, x_axis_config={"font_size": font_size}, x_length=width, y_length=height)

        # Create the KDE line plot with different colors for each batch
        kde = gaussian_kde(values, bw_method=bw_method)
        x_kde = np.linspace(0, num_bins, 1000)
        y_kde = kde(x_kde / num_bins) * max(hist_values) / max(kde(x_kde / num_bins))  # normalizing

        kde_line_points = [barchart.coords_to_point(x, y) for x, y in zip(x_kde, y_kde)]
        kde_line = VMobject()
        kde_line.set_points_as_corners(kde_line_points)

        min_label = (
            Text(f"Min: {values.min():.2f}", font_size=font_size)
            .next_to(barchart, direction=DOWN, buff=0.2)
            .align_to(barchart.get_corner(DOWN + LEFT), LEFT)
        )
        max_label = (
            Text(f"Max: {values.max():.2f}", font_size=font_size)
            .next_to(barchart, direction=DOWN, buff=0.2)
            .align_to(barchart.get_corner(DOWN + RIGHT), RIGHT)
        )

        group = Group(barchart, kde_line, min_label, max_label)
        group.move_to(UP * i * (height + 1))  # Adjust the vertical position based on the batch index

        group_list.append(group)

    return Group(*group_list)
