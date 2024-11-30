"""Create animation for changing v0 and T in the speed function.
to run:
manim -pqh change_v0_T.py
"""

from manim import *
import numpy as np

# https://pedestriandynamics.org/models/collision_free_speed_model/
velocity_eq = (
    MathTex(
        r"v(s) = \begin{cases} "
        + r"0 & 0 \leq s \leq l \\"
        + r"\min\left(\frac{s-l}{T}, v_0\right) & l < s \leq l + v_0T \\"
        + r"v_0 & s >  l + v_0T"
        + r"\end{cases}"
    )
    .to_corner(UP + RIGHT)
    .scale(0.7)
)


def setup_axes():
    """Set up the axes and add to the scene."""
    axes = Axes(
        x_range=[0, 10, 1],
        y_range=[0, 4, 1],
        axis_config={"tip_shape": StealthTip},
    )
    x_label = axes.get_x_axis_label("s [m]")
    y_label = axes.get_y_axis_label("v [m/s]")
    axes.add(x_label, y_label)
    return axes


def get_T_behavior_text(T_value):
    """Return the explanation based on the current value of T."""
    if T_value == 0.3:
        return "Agents have almost no time gap, behaving aggressively."
    elif T_value == 1.7:
        return "Agents maintain a large time gap, prioritizing safety."
    else:
        return ""


def get_V0_behavior_text(v0_value):
    """Return the explanation based on the current value of v0."""
    if v0_value == 2:
        return "Agents are running."
    elif v0_value == 0.5:
        return "Agents are slow, almost disinterested."
    else:
        return ""


T_values = [
    1,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.6,
    1.5,
    1.4,
    1.3,
    1.2,
    1.1,
    1,
]

v0_values = [
    1,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2,
    1.9,
    1.8,
    1.7,
    1.6,
    1.5,
    1.4,
    1.3,
    1.2,
    1.1,
    1.0,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]


class ChangingTAndV0(Scene):
    def construct(self):
        axes = setup_axes()
        # Initial values
        T = ValueTracker(1)
        v0 = ValueTracker(1)

        # Precise piecewise function with strict range
        def graph_func(s):
            if s > 6:
                return 0
            current_T = T.get_value()
            current_v0 = v0.get_value()
            if s <= 1:
                return 0
            elif 1 < s <= 1 + current_T * current_v0:
                return min(current_v0, (s - 1) / current_T)
            else:
                return current_v0

        graph = always_redraw(
            lambda: axes.plot(
                graph_func, x_range=[0, 6], use_smoothing=False, color=BLUE
            )
        )
        # Angle visualization
        filled_angle = always_redraw(
            lambda: Sector(
                arc_center=axes.c2p(1, 0),
                inner_radius=0,
                outer_radius=0.5,
                angle=np.arctan((v0.get_value()) / (T.get_value() * v0.get_value())),
                start_angle=0,
                color=BLUE,
                fill_opacity=0.5,
            )
        )
        angle = always_redraw(
            lambda: Angle(
                Line(
                    start=axes.c2p(1, 0), end=axes.c2p(1 + T.get_value(), 0)
                ),  # Dynamic base line
                Line(
                    start=axes.c2p(1, 0),
                    end=axes.c2p(1 + T.get_value() * v0.get_value(), v0.get_value()),
                ),  # Dynamic incline line
                radius=0.5,
                other_angle=False,
                color=ORANGE,
                dot=True,
            )
        )
        dot = always_redraw(lambda: Dot(axes.c2p(1, 0), color=WHITE))

        # Create T and v0 value texts
        t_text = always_redraw(
            lambda: MathTex(
                rf"\mathbf{{T = {T.get_value():.1f}\; [s]}}",
                font_size=24,
                color=RED if T.get_value() != 1 else WHITE,
            ).next_to(velocity_eq, DOWN, aligned_edge=LEFT, buff=0.8)
        )
        v0_text = always_redraw(
            lambda: MathTex(
                rf"\mathbf{{v_0 = {v0.get_value():.1f}\; [m/s]}}",
                font_size=24,
                color=RED if v0.get_value() != 1 else WHITE,
            ).next_to(velocity_eq, DOWN, aligned_edge=LEFT, buff=0.8)
        )

        # Add elements to the scene
        self.add(axes, velocity_eq)
        # Show equation and parameter texts
        self.play(Write(velocity_eq))

        # Animate changing T with smooth color transitions
        t_framebox = SurroundingRectangle(velocity_eq[0][24:25], buff=0.08, color=RED)

        self.play(Create(t_framebox))
        x_label0 = MathTex(r"l").next_to(axes.c2p(1, 0), DOWN)
        self.add(axes, x_label0)

        # Start graph animation
        self.play(Create(graph))
        self.play(Write(t_text))
        # self.add(angle, dot)

        explanation_text_T = always_redraw(
            lambda: Text(
                get_T_behavior_text(T.get_value()),
                font_size=20,
                font="Fira Code Symbol",
                color=YELLOW,
            ).next_to(t_text, DOWN, aligned_edge=LEFT, buff=0.2)
        )
        explanation_text_v0 = always_redraw(
            lambda: Text(
                get_V0_behavior_text(v0.get_value()),
                font_size=20,
                font="Fira Code Symbol",
                color=YELLOW,
            ).next_to(t_text, DOWN, aligned_edge=LEFT, buff=0.2)
        )
        dashed_line = always_redraw(
            lambda: DashedLine(
                start=axes.c2p(1 + v0.get_value() * T.get_value(), v0.get_value()),
                end=axes.c2p(
                    1 + v0.get_value() * T.get_value(), graph_func(1)
                ),  # End at (4, graph_func(4))
                dash_length=0.1,  # Adjust dash length
                color=BLUE,
            )
        )

        # Add the dashed line to the scene
        self.play(Create(dashed_line))
        self.play(Create(filled_angle))
        self.play(Create(angle))
        self.add(explanation_text_T, explanation_text_v0)
        for target_T in T_values:
            self.play(
                T.animate.set_value(target_T).set_color(RED),
                run_time=0.1,
                rate_func=smooth,
            )
            if target_T == min(T_values) or target_T == max(T_values):
                self.wait(5)

        # ================================ v0 ==================
        # Reset T color and transition to v0
        # self.play(T.animate.set_color(WHITE))
        self.play(FadeOut(t_text), FadeOut(explanation_text_T))  # Remove T text
        v0_framebox = SurroundingRectangle(velocity_eq[0][38], buff=0.15, color=RED)
        self.play(ReplacementTransform(t_framebox, v0_framebox))

        # Transition to v0

        self.play(Write(v0_text))  # Add v0 text

        # Animate changing v0
        for target_v0 in v0_values:
            self.play(
                v0.animate.set_value(target_v0).set_color(RED),
                run_time=0.1,
                rate_func=smooth,
            )
            if target_v0 == min(v0_values) or target_v0 == max(v0_values):
                self.wait(5)

        # Reset v0 color and hold
        self.play(v0.animate.set_color(WHITE))
        self.wait(1)

        # Fade out scene elements
        self.play(
            FadeOut(axes),
            FadeOut(graph),
            FadeOut(velocity_eq),
            FadeOut(v0_text),
            FadeOut(t_framebox),
            FadeOut(v0_framebox),
            FadeOut(dashed_line),
            FadeOut(angle),
            FadeOut(filled_angle),
            FadeOut(x_label0),
        )
