import inspect
from manim import UP, DOWN, LEFT, RIGHT, DEGREES, Arrow, Mobject, VMobject, LineJointType, FadeIn


class ConnectionArrow(Mobject):
    """
    connect two other objects with an arrow
    (mobject_from, edge_from)
    (mobject_to, edge_to)
    connect a line perpendicular to the from edge of the from mobject
    and draw an arrow perpendicular to that line, pointing to the to edge of the to mobject

    e.g.,
    connect_arrow = ConnectArrow(from=(mobject_from, edge_from), to=(mobject_to, edge_to))
    """

    def __init__(self, *args, **kwargs):
        self.mobject_from, self.edge_from = kwargs.get("_from")
        self.mobject_to, self.edge_to = kwargs.get("to")
        self.buff = kwargs.get("buff", 0.1)
        self.stroke_width = kwargs.get("stroke_width", 3)
        # ================== pop kwargs for parent ==================
        parent_params = inspect.signature(super().__init__).parameters
        [kwargs.pop(kw) for kw in list(kwargs.keys()) if kw not in parent_params]
        super().__init__(*args, **kwargs)
        # ===========================================================s
        self._check_validity()
        self._make_straight()

    def _check_validity(self):
        # Conditions for edge_from
        cond_from_up_down = tuple(self.edge_from) in {tuple(UP), tuple(DOWN)}
        cond_from_left_right = tuple(self.edge_from) in {tuple(LEFT), tuple(RIGHT)}

        # Conditions for edge_to
        cond_to_up_down = tuple(self.edge_to) in {tuple(UP), tuple(DOWN)}
        cond_to_left_right = tuple(self.edge_to) in {tuple(LEFT), tuple(RIGHT)}

        # Final combined condition
        valid_combination = (cond_from_up_down and cond_to_left_right) or (cond_from_left_right and cond_to_up_down)
        assert valid_combination, "Invalid edge combination"

    def _make_curved(self):
        """
        Method to create the ConnectionArrow shape.
        """
        start_point = self.mobject_from.get_edge_center(self.edge_from)
        end_point = self.mobject_to.get_edge_center(self.edge_to)

        # Determine path_arc based on edge_from direction
        if tuple(self.edge_from) == tuple(UP):
            path_arc = -90 * DEGREES
        elif tuple(self.edge_from) == tuple(DOWN):
            path_arc = 90 * DEGREES
        elif tuple(self.edge_from) == tuple(LEFT):
            path_arc = 0 * DEGREES
        elif tuple(self.edge_from) == tuple(RIGHT):
            path_arc = 180 * DEGREES

        arrow = Arrow(start_point, end_point, path_arc=path_arc)

        self.add(arrow)

    def _make_straight(self):
        # 1. Calculate the turning point coordinate
        turning_point = None

        start_point = self.mobject_from.get_edge_center(self.edge_from)
        end_point = self.mobject_to.get_edge_center(self.edge_to)
        start_point += self.buff * self.edge_from
        end_point += self.buff * self.edge_to

        if tuple(self.edge_from) in {tuple(UP), tuple(DOWN)}:
            x_coord = start_point[0]
            y_coord = end_point[1]
        elif tuple(self.edge_from) in {tuple(LEFT), tuple(RIGHT)}:
            y_coord = start_point[1]
            x_coord = end_point[0]

        turning_point = [x_coord, y_coord, 0]

        # This will contain our path (a line that turns and goes to just before the end_point)
        path = VMobject().set_points_as_corners([start_point, turning_point, end_point - 0.05 * (end_point - turning_point)])
        path.joint_type = LineJointType.ROUND

        # This is a short arrow just to show the tip at the end
        arrow_tip = Arrow(
            turning_point,
            end_point,
            buff=0,
            stroke_width=self.stroke_width,
            max_stroke_width_to_length_ratio=0.1,
        )

        self.add(path, arrow_tip)

    def on_create(self):
        transforms = [FadeIn(self)]
        return transforms
