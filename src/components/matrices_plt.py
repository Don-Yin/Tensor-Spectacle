from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from manim import SVGMobject, VGroup, VMobject, Text, DOWN
import os
from uuid import uuid4


def plot_batch(batch, idx, save_path, alpha=0.5):
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


def create_3D_matrix(tensor, distance=0.1, use_opengl_renderer=False):
    if use_opengl_renderer:
        from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup as VGroup, OpenGLVMobject as VMobject

    assert len(tensor.shape) == 4, "Input tensor must be 4D"

    save_path = Path("media", "image_cache")
    save_path.mkdir(parents=True, exist_ok=True)
    [os.remove(file) for file in save_path.glob("*")]

    num_batches = tensor.shape[0]
    hexes = [str(uuid4().hex) for _ in range(num_batches)]

    for i in range(num_batches):
        plot_batch(tensor[i], hexes[i], save_path=save_path)

    svg_objects = [SVGMobject(save_path / f"3d_plot_{hex}.svg") for hex in hexes]

    vgroup = VGroup()
    previous_object = None

    for i, obj in enumerate(svg_objects):
        shape_label = Text(f"{tensor[i].shape}").scale_to_fit_width(obj.get_width()).next_to(obj, direction=DOWN, buff=0.1)
        batch_group = VGroup(obj, shape_label)
        if previous_object:
            batch_group.next_to(previous_object, direction=DOWN, buff=distance)
        vgroup.add(batch_group)
        previous_object = batch_group

    return vgroup
