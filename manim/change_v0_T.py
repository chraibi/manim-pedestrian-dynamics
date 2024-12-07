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


def get_T_behavior_text(T_value, Tmin, Tmax):
    """Return the explanation based on the current value of T."""
    if T_value == Tmin:
        return "Agents have almost no time gap, behaving aggressively."
    elif T_value == Tmax:
        return "Agents maintain a large time gap, prioritizing safety."
    else:
        return ""


def get_V0_behavior_text(v0_value, vmin, vmax):
    """Return the explanation based on the current value of v0."""
    if v0_value == vmax:
        return "Agents are running."
    elif v0_value == vmin:
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
    0.2,
    0.1,
    0.2,
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
    1.8,
    1.9,
    2.0,
    1.9,
    1.8,
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


def bounce_out(t):
    """
    A bounce out rate function that creates a bouncing effect.
    Borrowed from Manim's animation utilities.
    """
    a = 0.1
    if t < 4 / 11.0:
        return a * (121 * t * t / 16.0)
    elif t < 8 / 11.0:
        return a * (363 / 40.0 * t * t - 99 / 10.0 * t + 17 / 5.0)
    elif t < 9 / 10.0:
        return a * (4356 / 361.0 * t * t - 35442 / 1805.0 * t + 16061 / 1805.0)
    return a * (54 / 5.0 * t * t - 513 / 25.0 * t + 268 / 25.0)


class ChangingTAndV0(Scene):
    def visualize_agent_diameter(self, axes):
        """Visualize the agent diameter as a circle and animate it."""
        circle_radius = 1  # Assuming l = 1 diameter
        circle = Circle(radius=circle_radius, color=BLUE, fill_opacity=0.3)
        semicircle = Arc(radius=1, angle=PI, color=GREEN, fill_opacity=0.3)

        dot = Dot()
        self.add(dot)
        self.play(GrowFromCenter(circle))

        agent_label = MathTex(
            r"\text{Agent is a circle with diameter } l", font_size=24
        ).next_to(dot, UP * 5)

        self.add(agent_label)
        dot2 = dot.copy().shift(RIGHT).set_color(BLUE)
        dot3 = dot2.copy().set_color(BLUE)

        self.play(Transform(dot, dot2))
        self.play(MoveAlongPath(dot2, semicircle), run_time=1, rate_func=linear)
        line = Line(dot3, dot2)
        diameter_label = MathTex(r"l", font_size=28).next_to(line, DOWN)
        self.add(line)
        self.play(Write(diameter_label))
        self.wait(3)
        self.play(
            FadeOut(agent_label),
            diameter_label.animate.next_to(axes.c2p(1, 0), DOWN),
            FadeOut(line),
            FadeOut(dot),
            FadeOut(dot2),
            FadeOut(dot3),
            circle.animate.move_to(axes.c2p(1, 0))
            .scale(0.05)
            .set_fill(opacity=1)
            .set_color(WHITE),
        )

    def construct(self):
        axes = setup_axes()
        # Initial values
        T = ValueTracker(1)
        v0 = ValueTracker(1)
        axes = setup_axes()
        self.visualize_agent_diameter(axes)

        # 1 Show circle and explain l. l then moves to the axis.
        # 2 Show equation.
        # 3. plot the graph of the equation including the dashed line and the angles.
        # 4 start with changing T and then v.
        # 5 fadeout all elements
        # TODO reduce the size of the function control.
        # TODO refactor function show equation
        # TODO refactor function visualize T
        # TODO refactor function visualize v0
        # fadeout all elements in one function. Use *args or something so that the arguments are not explicit.
        # some functions have dependencies. for example vis T and vis v0. entangle them.
        # --------------
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
                rf"T = {T.get_value():.1f}\; [s]",
                font_size=24,
                color=RED if T.get_value() != 1 else WHITE,
            ).next_to(velocity_eq, DOWN, aligned_edge=LEFT, buff=0.8)
        )
        v0_text = always_redraw(
            lambda: MathTex(
                rf"v_0 = {v0.get_value():.1f}\; [m/s]",
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
        t_in_equation = velocity_eq[0][24:25]
        moving_t = MathTex(r"T", font_size=24).move_to(t_in_equation.get_center())
        self.play(FocusOn(t_in_equation))
        self.play(Create(t_framebox))
        self.play(
            t_in_equation.animate.set_color(YELLOW),  # Highlight the T in equation
            moving_t.animate.move_to(t_text[0][0].get_center()),
            run_time=1,
        )
        self.play(Write(t_text))
        # Start graph animation
        self.play(Create(graph))
        explanation_text_T = always_redraw(
            lambda: Text(
                get_T_behavior_text(
                    T.get_value(), Tmin=min(T_values), Tmax=max(T_values)
                ),
                font_size=20,
                font="Fira Code Symbol",
                color=YELLOW,
            ).next_to(t_text, DOWN, aligned_edge=LEFT, buff=0.2)
        )
        explanation_text_v0 = always_redraw(
            lambda: Text(
                get_V0_behavior_text(
                    v0.get_value(), vmin=min(v0_values), vmax=max(v0_values)
                ),
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
        self.play(FadeOut(t_text), FadeOut(explanation_text_T), FadeOut(moving_t))
        self.play(t_in_equation.animate.set_color(WHITE))
        v0_framebox = SurroundingRectangle(velocity_eq[0][38:40], buff=0.15, color=RED)
        v0_in_equation = velocity_eq[0][38:40]
        moving_v0 = MathTex(r"v_0", font_size=24).move_to(v0_in_equation.get_center())
        self.play(FocusOn(v0_in_equation))
        self.play(ReplacementTransform(t_framebox, v0_framebox))
        # Transition to v0
        self.play(
            v0_in_equation.animate.set_color(YELLOW),  # Highlight the v0 in equation
            moving_v0.animate.move_to(v0_text[0][0:2].get_center()),
            run_time=1,
        )

        self.play(Write(v0_text))  # Add v0 text
        self.play(FadeOut(moving_v0), v0_in_equation.animate.set_color(WHITE))

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
        )
