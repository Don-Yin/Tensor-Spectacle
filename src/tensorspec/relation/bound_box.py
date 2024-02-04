from manim import Group, Mobject, DL, UR, Tex, Rectangle, BLACK, WHITE, UP, DOWN


def bounding_moject(mobject: Mobject, label: str, font_size=32):
    """
    return the bounding box as a mobject with label
    """
    box = (
        Rectangle(
            height=mobject.height + 1,
            width=mobject.width + 0.3,
            fill_opacity=0,
            fill_color=BLACK,
            color=WHITE,
            stroke_width=3,
            stroke_color=BLACK,
        )
        .move_to(mobject.get_center())
        .shift(DOWN * 0.1)
    )

    label = Tex(label, color=BLACK, font_size=font_size).next_to(box, UP * 2)

    return Group(box, label)
