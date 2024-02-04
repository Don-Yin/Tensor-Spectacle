from manim import Group, Mobject, DL, UR


def repeat_mobject(mobject: Mobject, num_repeats: int, distance: float) -> Group:
    """Repeats a mobject num_repeats times, with distance between each repeat"""
    mobjects = [mobject.copy().shift(DL, i * distance) for i in range(num_repeats)]
    mobjects.reverse()
    group = Group(*mobjects)
    return group


# self.masks_obj_list = [
# Rectangle(
#     height=self.dimensions_2d[0],
#     width=self.dimensions_2d[1],
#     grid_xstep=1.0,
#     grid_ystep=1.0,
#     fill_opacity=0,
#     fill_color=BLACK,
#     color=WHITE,
#     stroke_width=self.stroke_width,
# )
# .set_stroke(width=self.stroke_width)
# .scale_to_fit_width(self.rgb_mobjects_list[i].width)
# .move_to(self.rgb_mobjects_list[i].get_center())
# for i in range(self.num_channels)
# ]

# self.data = []
# for i in range(self.num_channels):
# masked_channel = Group(self.rgb_mobjects_list[i], self.masks_obj_list[i])
# self.data.append(masked_channel)
# self.data = Group(*self.data)
# for i, channel in enumerate(self.data):
# channel.shift(i * self.between_channel_distance * DL)
