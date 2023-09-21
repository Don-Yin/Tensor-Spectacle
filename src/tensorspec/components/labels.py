from manim import Rectangle, Text, VGroup, WHITE, BLUE, RIGHT


def make_labels(labels: list[str], selected_idx=0, use_opengl_renderer=False):
    """make a list of horizontally aligned rectangles from left to right"""
    global VGroup
    if use_opengl_renderer:
        from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup as VGroup

    rectangles = [Rectangle(color=WHITE, fill_opacity=1, width=0.2, height=0.1) for i in range(len(labels))]
    rectangles[selected_idx].set_color(BLUE)

    # add a text to the selected rectangle
    text = Text(labels[selected_idx]).scale(0.2)

    # Adjust the height and width of the selected rectangle to match that of the text object
    rectangles[selected_idx] = Rectangle(color=BLUE, fill_opacity=1, width=text.get_width(), height=0.2)
    text.move_to(rectangles[selected_idx].get_center())
    rectangles[selected_idx].add(text)

    return VGroup(*rectangles).arrange(RIGHT, buff=0.1).scale(1.5)
