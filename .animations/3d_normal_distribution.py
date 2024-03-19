"""
https://www.t-ott.dev/2021/11/24/animating-normal-distributions
"""

import toml

from manim import (
    ThreeDScene,
    ThreeDAxes,
    ValueTracker,
    MathTex,
    always_redraw,
    Surface,
    rate_functions,
    DEGREES,
    UP,
    Uncreate,
    WHITE,
    tempconfig,
    Create,
    rgb_to_color,
)

from manim.utils.file_ops import open_file as open_media_file
import math

COLOR_RAMP = [
    rgb_to_color([57 / 255, 0.0, 153 / 255]),
    rgb_to_color([158 / 255, 0.0, 89 / 255]),
    rgb_to_color([1.0, 0.0, 84 / 255]),
    rgb_to_color([1.0, 84 / 255, 0.0]),
    rgb_to_color([1.0, 189 / 255, 0.0]),
]


def PDF_bivariate_normal(x_1, x_2, params):
    """
    Probability density function of a mixture of three bivariate normal distributions
    """
    result = 0
    for mu_1, mu_2, sigma_1, sigma_2, rho in params:
        normalizing_const = 1 / (2 * math.pi * sigma_1 * sigma_2 * math.sqrt(1 - rho**2))
        exp_coeff = -(1 / (2 * (1 - rho**2)))
        A = ((x_1 - mu_1) / sigma_1) ** 2
        B = -2 * rho * ((x_1 - mu_1) / sigma_1) * ((x_2 - mu_2) / sigma_2)
        C = ((x_2 - mu_2) / sigma_2) ** 2
        result += normalizing_const * math.exp(exp_coeff * (A + B + C))

    return result / len(params)  # Average over the number of distributions


class Example(ThreeDScene):
    """
    Scene plots the surface of the probability density function of the bivariate
    normal distribution, then animates various adjustments to the means (mu) of
    x_1 and x_2
    """

    def construct(self):
        self.camera.background_color = WHITE
        # ====================================
        ax = ThreeDAxes(x_range=[-5, 5, 1], y_range=[-5, 5, 1], z_range=[0, 0.2, 0.1])
        x_label = ax.get_x_axis_label(r"x_1")
        y_label = ax.get_y_axis_label(r"x_2", edge=UP, buff=0.2)
        z_label = ax.get_z_axis_label(r"\phi(x_1, x_2)", buff=0.2)

        # Initialize ValueTrackers to adjust means
        mu_1_a = ValueTracker(0)
        mu_2_a = ValueTracker(0)
        mu_1_b = ValueTracker(0)
        mu_2_b = ValueTracker(0)
        # mu_1_c = ValueTracker(0)
        # mu_2_c = ValueTracker(0)

        # value adjustments to mu_1 and mu_2
        distribution = always_redraw(
            lambda: Surface(
                lambda u, v: ax.c2p(
                    u,
                    v,
                    PDF_bivariate_normal(
                        u,
                        v,
                        [
                            (mu_1_a.get_value(), mu_2_a.get_value(), 1, 1, 0),
                            (mu_1_b.get_value(), mu_2_b.get_value(), 1, 1, 0),
                            # (mu_1_c.get_value(), mu_2_c.get_value(), 1, 1, 0),
                        ],
                    ),
                ),
                resolution=(42, 42),
                u_range=[-4.5, 4.5],
                v_range=[-4.5, 4.5],
                fill_opacity=0.7,
            ).set_fill_by_value(
                axes=ax,
                # Utilize color ramp colors with, higher values are "warmer"
                colors=[
                    (COLOR_RAMP[0], 0),
                    (COLOR_RAMP[1], 0.05),
                    (COLOR_RAMP[2], 0.1),
                    (COLOR_RAMP[3], 0.15),
                    (COLOR_RAMP[4], 0.2),
                ],
            )
        )

        # Set up animation
        self.set_camera_orientation(theta=-90 * DEGREES, phi=0, frame_center=[0, 0, 0], zoom=0.5)
        self.play(Create(distribution))
        self.wait()
        self.move_camera(theta=-70 * DEGREES, phi=70 * DEGREES, frame_center=[0, 0, 2], zoom=0.6, run_time=2)
        # Begin animation
        self.play(
            mu_1_a.animate.set_value(-2),
            mu_2_a.animate.set_value(-2),
            mu_1_b.animate.set_value(2),
            mu_2_b.animate.set_value(2),
            # mu_1_c.animate.set_value(-2),
            # mu_2_c.animate.set_value(2),
            run_time=2,
            rate_func=rate_functions.smooth,
        )
        self.wait()
        # top down view
        self.move_camera(theta=-90 * DEGREES, phi=0, frame_center=[0, 0, 0], zoom=0.5, run_time=2)
        self.wait()


def test_example_runs():
    with open("CONFIG.toml", "r") as toml_file:
        config_data = toml.load(toml_file)

    with tempconfig(config_data):
        scene = Example()
        scene.render()
        open_media_file(scene.renderer.file_writer.movie_file_path)


if __name__ == "__main__":
    test_example_runs()
