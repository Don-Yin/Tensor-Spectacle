"""
Script making the stack of 3D cubes at the tensor -> 3D matrix section using matplotlib to generate the svg and use those as mobjects. This is much faster than using manim native cubes.
"""

from manim import SVGMobject, VGroup, VMobject, Text, DOWN, Mobject
import matplotlib.pyplot as plt
from uuid import uuid4
from pathlib import Path
import numpy as np
import os
from torch import tensor


def plot_channel(batch, idx, save_path, alpha=0.5):
    """
    Plots a 3D representation of data from a batch using voxels and saves it as an SVG.
    Note: using a tensor with channel size less than 3 is recommanded.
    Parameters:
        - batch (numpy.ndarray): A 3D numpy array representing the data to be plotted.
        - idx (int): An identifier to label the saved plot (e.g., plot number or iteration).
        - save_path (Path or str): The directory where the SVG plot should be saved.
        - alpha (float, optional): The opacity of the voxel colors. Default is 0.5.

    Description:
        The function visualizes the data in a 3D space, color-mapping the values to the seismic colormap.
        The function saves the 3D voxel representation in a dark theme as an SVG to the specified path.
    """
    plt.style.use("dark_background")
    fig = plt.figure()

    batch = batch.numpy()
    max_val = np.max(batch)
    min_val = np.min(batch)
    axes = list(batch.shape)
    traj = np.random.choice([-1, 1], axes)
    normalized_values = (batch - min_val) / (max_val - min_val)

    colors = np.empty(axes + [4], dtype=np.float32)
    colors[..., :3] = plt.cm.seismic(normalized_values)[..., :3]
    colors[..., 3] = alpha

    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(traj, facecolors=colors, edgecolors="black", linewidth=0.1)
    ax.set_box_aspect([np.ptp(arr) for arr in [range(axes[0]), range(axes[1]), range(axes[2])]])
    ax.view_init(elev=20, azim=-30)
    ax.set_axis_off()

    plt.tight_layout()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    fig.savefig(save_path / f"3d_plot_{idx}.svg", format="svg", transparent=True, bbox_inches=0, pad_inches=0)
    plt.close(fig)


class Matrix3DMatplotlib(Mobject):
    def __init__(self, tensor: tensor, use_opengl_renderer: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert tensor.dim() == 4, f"The tensor must be 4-dimensional (batch, c, h, w) got {tensor.dim()}"
        self.tensor = tensor
        self.distance = 0.1
        global VGroup, VMobject
        if use_opengl_renderer:
            from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup as VGroup, OpenGLVMobject as VMobject
        self.make_matrix()

    def make_matrix(self):
        save_path = Path("media", "image_cache")
        save_path.mkdir(parents=True, exist_ok=True)
        [os.remove(file) for file in save_path.glob("*")]

        num_channels = self.tensor.shape[0]
        hexes = [str(uuid4().hex) for _ in range(num_channels)]

        for i in range(num_channels):
            plot_channel(self.tensor[i], hexes[i], save_path=save_path)

        svg_objects = [SVGMobject(save_path / f"3d_plot_{hex}.svg") for hex in hexes]
        vgroup = VGroup()
        previous_object = None

        for i, obj in enumerate(svg_objects):
            shape_label = Text(f"{self.tensor[i].shape}").scale_to_fit_width(obj.get_width())
            shape_label = shape_label.next_to(obj, direction=DOWN, buff=0.1)
            batch_group = VGroup(obj, shape_label)
            if previous_object:
                batch_group.next_to(previous_object, direction=DOWN, buff=self.distance)
            vgroup.add(batch_group)
            previous_object = batch_group
        self.add(vgroup)
