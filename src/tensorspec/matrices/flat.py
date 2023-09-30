from pathlib import Path
from manim import (
    Rectangle,
    Group,
    Text,
    Mobject,
    UP,
    DL,
    BLACK,
    WHITE,
    VGroup,
    ImageMobject,
    FadeIn,
    Create,
    Write,
)
import inspect
from torch.nn import functional as F
import numpy as np
from PIL import Image


class FlatMatrix3DBase(Mobject):
    def __init__(self, **kwargs):
        """dimensions: the final dimensions to which the image is resized; the channels will be interpolated"""
        self.dimensions, self.label = kwargs.get("dimensions"), kwargs.get("label")
        # ================== pop kwargs for parent ==================
        parent_params = inspect.signature(super().__init__).parameters
        [kwargs.pop(kw) for kw in list(kwargs.keys()) if kw not in parent_params]
        super().__init__(**kwargs)  # pop and feed to super; mobject doesn't take dimensions as an arg; only in base
        # ===========================================================
        assert len(self.dimensions) == 3, "dimensions must be a tuple of length 3"

        self.num_channels, self.dimensions_2d = self.dimensions[0], self.dimensions[1:]
        self.total_channel_thickness = self.get_channel_thickness(self.num_channels)
        self.stroke_width = self.get_stroke_width(self.num_channels)
        self.between_channel_distance = self.total_channel_thickness / (self.num_channels - 1)

    def get_stroke_width(self, channels: int):
        """dynamic stroke width based on the number of channels"""
        limit_range = (0.8, 1.3)
        if channels <= 3:
            return limit_range[1]
        if channels >= 64:
            return limit_range[0]
        base = (limit_range[1] / limit_range[0]) ** (1 / 61)  # 61 is the difference between 64 and 3
        decay_factor = 64 - channels
        stroke_width = limit_range[1] / (base**decay_factor)
        return stroke_width

    def get_channel_thickness(self, channels: int):
        limit_range = (1.2, 2)
        if channels <= 3:
            return limit_range[0]
        if channels >= 64:
            return limit_range[1]
        base = (limit_range[0] / limit_range[1]) ** (1 / 61)  # 61 is the difference between 64 and 3
        decay_factor = 64 - channels
        channel_thickness = limit_range[0] / (base**decay_factor)
        return channel_thickness


def channel_interpolate(tensor, out_channels):
    """
    Interpolates a 3D tensor along the channel axis.
    """
    tensor = tensor.permute(1, 0, 2)  # Bring channels to the middle
    tensor = F.interpolate(tensor.unsqueeze(0), [out_channels, tensor.size(2)], mode="bilinear")
    tensor = tensor.squeeze(0).permute(1, 0, 2)
    return tensor


class FlatMatrix3DImage(FlatMatrix3DBase):
    """
    IMPORTANT: too much pain to decode the image; just paste it in as a png; right now this can't be used in transformations
    """

    def __init__(self, **kwargs):
        """additional params: image_path: Path"""
        super().__init__(**kwargs)
        self.image_path = kwargs.get("image_path")
        assert isinstance(self.image_path, Path), "image_path must be a Path object"
        self.make_matrix()

    def make_matrix(self):
        """
        1. read the image; if the image shape is not (c, h, w), then transpose it to (c, h, w); convert to rgb
        2. resize the image to the self.dimensions_2d, which is a tuple of (height, width)
        3. convert to a tensor
        the order of the above doesn't matter, the goal is to get a tensor of shape (c, h, w) of the resized image
        4. if the image channel is smaller than the self.num_channels, interpolate it to the self.num_channels
        """
        image_tensor = Image.open(self.image_path).convert("RGB")
        image_tensor = image_tensor.resize((2 * i for i in self.dimensions_2d))
        image_tensor = np.array(image_tensor)
        # resize to the self.dimensions_2d

        rgb_data = [image_tensor[:, :, i] for i in range(image_tensor.shape[-1])]
        self.rgb_mobjects_list = [
            ImageMobject(rgb_data[i]).scale_to_fit_width(self.dimensions_2d[0]) for i in range(image_tensor.shape[-1])
        ]
        self.masks_obj_list = [
            Rectangle(
                height=self.dimensions_2d[0],
                width=self.dimensions_2d[1],
                grid_xstep=1.0,
                grid_ystep=1.0,
                fill_opacity=0,
                fill_color=BLACK,
                color=WHITE,
                stroke_width=self.stroke_width,
            )
            .set_stroke(width=self.stroke_width)
            .scale_to_fit_width(self.rgb_mobjects_list[i].width)
            .move_to(self.rgb_mobjects_list[i].get_center())
            for i in range(3)
        ]

        self.data = []
        for i in range(3):
            masked_channel = Group(self.rgb_mobjects_list[i], self.masks_obj_list[i])
            self.data.append(masked_channel)
        self.data = Group(*self.data)
        for i, channel in enumerate(self.data):
            channel.shift(i * self.between_channel_distance * DL)

        self.text_obj = Text(self.label, color=WHITE)
        self.text_obj = self.text_obj.next_to(self.data, UP, buff=self.data.height * 0.1).scale_to_fit_width(
            self.data.width * 0.05 * len(self.label)
        )

        self.add(self.data, self.text_obj)

    def on_create(self):
        animations = []
        for masked_channel in self.data:
            for mobject in masked_channel:
                animations.append(FadeIn(mobject))
        animations.append(Write(self.text_obj))
        return animations


class FlatMatrix3D(FlatMatrix3DBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.make_matrix()

    def make_matrix(self):
        self.matrix = VGroup(
            *[
                Rectangle(
                    height=self.dimensions_2d[0],
                    width=self.dimensions_2d[1],
                    grid_xstep=1.0,
                    grid_ystep=1.0,
                    fill_opacity=1,
                    fill_color=BLACK,  # block
                    color=WHITE,  # lines
                    stroke_width=self.stroke_width,
                ).set_stroke(width=self.stroke_width)
                for _ in range(self.num_channels)
            ]
        )
        for i, rect in enumerate(self.matrix):
            rect.shift(i * self.between_channel_distance * DL)

        self.dimension_label = Text(str(self.dimensions), font_size=8, color=WHITE)
        self.dimension_label = self.dimension_label.next_to(self.matrix, UP, buff=self.matrix.height * 0.1).scale_to_fit_width(
            self.matrix.width * 0.03 * len(str(self.dimensions))
        )

        self.text_str = Text(self.label, color=WHITE)
        self.text_str = self.text_str.next_to(self.dimension_label, UP, buff=self.matrix.height * 0.1).scale_to_fit_width(
            self.matrix.width * 0.05 * len(self.label)
        )

        self.add(self.matrix, self.text_str, self.dimension_label)

    def on_create(self):
        animations = []
        for rect in self.matrix:
            animations.append(Create(rect))
        animations.append(Write(self.text_str))
        animations.append(Write(self.dimension_label))
        return animations

    def _resize_label(self, width):
        self.text_str.scale_to_fit_width(width)
        return self

    def _hide_dimensions(self):
        self.dimension_label.set_opacity(0)
        return self
