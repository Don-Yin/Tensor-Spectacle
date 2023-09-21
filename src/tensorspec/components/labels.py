from manim import Rectangle, Text, VGroup, WHITE, BLUE, RIGHT


def make_labels(labels: list[str], selected_idx=0, use_opengl_renderer=False):
    """
    Creates a list of horizontally aligned rectangles with one selected rectangle having a label.
        Parameters:
        - labels (list[str]): A list of strings where each string is intended to be a label for a rectangle.
        - selected_idx (int, optional): The index of the rectangle in the list that is to be colored differently and labeled. Default is 0.
        - use_opengl_renderer (bool, optional): Whether or not the OpenGL renderer is being used. If True, uses OpenGL-specific classes. Default is False.
    """
    global VGroup  # a workaround for opengl
    if use_opengl_renderer:
        from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup as VGroup
    rectangles = [Rectangle(color=WHITE, fill_opacity=1, width=0.2, height=0.1) for i in range(len(labels))]
    rectangles[selected_idx].set_color(BLUE)
    text = Text(labels[selected_idx]).scale(0.2)
    rectangles[selected_idx] = Rectangle(color=BLUE, fill_opacity=1, width=text.get_width(), height=0.2)
    text.move_to(rectangles[selected_idx].get_center())
    rectangles[selected_idx].add(text)
    return VGroup(*rectangles).arrange(RIGHT, buff=0.1).scale(1.5)
