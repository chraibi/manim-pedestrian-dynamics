"""
Manim animation for visualizing speed function parameters.
Run with: manim -pqh change_v0_T.py
"""

from manim import *
import numpy as np

# Velocity equation parameters
T_values = [
    1,
    # 0.8,
    # 0.7,
    # 0.6,
    # 0.5,
    # 0.4,
    # 0.3,
    # 0.2,
    # 0.1,
    # 0.2,
    # 0.3,
    # 0.4,
    # 0.5,
    # 0.6,
    # 0.7,
    # 0.8,
    # 0.9,
    # 1,
    # 1.1,
    # 1.2,
    # 1.3,
    # 1.4,
    # 1.5,
    # 1.6,
    # 1.7,
    # 1.8,
    # 1.9,
    # 2.0,
    # 1.9,
    # 1.8,
    # 1.7,
    # 1.6,
    # 1.5,
    # 1.4,
    # 1.3,
    # 1.2,
    # 1.1,
    # 1,
]
v0_values = [
    1,
    # 1.1,
    # 1.2,
    # 1.3,
    # 1.4,
    # 1.5,
    # 1.6,
    # 1.7,
    # 1.8,
    # 1.9,
    # 2,
    # 1.9,
    # 1.8,
    # 1.7,
    # 1.6,
    # 1.5,
    # 1.4,
    # 1.3,
    # 1.2,
    # 1.1,
    # 1.0,
    # 0.9,
    # 0.8,
    # 0.7,
    # 0.6,
    # 0.5,
    # 0.6,
    # 0.7,
    # 0.8,
    # 0.9,
    # 1.0,
]


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


def generate_non_overlapping_positions(
    base_position, direction_norm, num_agents=4, seed=42, min_distance=1.0
):
    """
    Generate non-overlapping agent positions

    Parameters:
    - base_position: Center point to base positions around
    - direction_norm: Normalized direction vector
    - num_agents: Number of agents to position
    - seed: Random seed for reproducibility
    - min_distance: Minimum distance between agents

    Returns:
    List of agent positions
    """
    # Set a fixed random seed
    np.random.seed(seed)

    positions = []
    max_attempts = 100

    while len(positions) < num_agents:
        # Base distance along movement direction with some randomness
        base_distance = np.random.uniform(1.5, 3.0)
        base_pos = base_position + direction_norm * base_distance

        # Add random perpendicular offset
        # Create a perpendicular vector by rotating direction_norm
        perp_vector = np.array([-direction_norm[1], direction_norm[0], 0])
        offset = perp_vector * np.random.uniform(-1.5, 1.5)

        candidate_pos = base_pos + offset

        # Check for overlap with existing positions
        if not any(
            np.linalg.norm(candidate_pos - existing_pos) < min_distance
            for existing_pos in positions
        ):
            positions.append(candidate_pos)

        # Prevent infinite loop
        if len(positions) >= num_agents or max_attempts <= 0:
            break
        max_attempts -= 1

    return positions


def get_new_line(line):
    # Explicitly set the start point to the exact coordinates of the original line
    original_start = line.get_start()

    # Calculate the original length
    original_length = np.linalg.norm(line.get_end() - line.get_start())

    # Calculate the original direction
    original_direction = (line.get_end() - line.get_start()) / np.linalg.norm(
        line.get_end() - line.get_start()
    )

    # Rotation matrix for slight downward angle
    angle = -np.pi / 30
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [1, 0, 0],
        ]
    )
    # Rotate the direction
    new_direction = np.dot(rotation_matrix, original_direction)
    # Calculate new endpoint while maintaining original length
    new_end = original_start + new_direction * original_length

    # Create new arrow with explicitly set start point
    new_line = Arrow(start=original_start, end=new_end, color=YELLOW)

    return new_line


class ChangingTAndV0(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configuration flags to control visualization stages
        self.config = {
            "agentExit": True,
            "show_agent_diameter": False,
            "show_equation": False,
            "animate_t_parameter": False,
            "animate_v0_parameter": False,
        }

    def AgentExitVisualization(self, A=1, D=1):
        # 1. Agent appears
        agent_circle = Circle(radius=0.5, color=BLUE, fill_opacity=0.5)
        agent_label = (
            MathTex(r"i", color=WHITE)
            .scale(0.7)
            .move_to(agent_circle.get_center() + LEFT * 0.1)
        )
        self.play(GrowFromCenter(agent_circle), Create(agent_label))
        # 2. Exit icon appears
        exit_icon = Triangle(fill_color=RED, fill_opacity=0.1, stroke_color=RED)
        exit_icon.scale(0.3).next_to(agent_circle, RIGHT, buff=4.5)
        self.play(Create(exit_icon), run_time=1)
        self.wait(1)

        # 3. Directional line (walking direction) appears
        direction = exit_icon.get_center() - agent_circle.get_center()
        direction_to_exit = direction / np.linalg.norm(direction)
        dashed_line = DashedLine(
            start=agent_circle.get_center(),
            end=exit_icon.get_center(),
            color=GREEN,
            stroke_width=2,
        )
        arrow_to_exit = Line(
            start=agent_circle.get_center(),
            end=agent_circle.get_center() + direction_to_exit,
            color=YELLOW,
        )
        arrow_to_exit.add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        e0_label = MathTex(r"\overrightarrow{e_0}", color=WHITE).scale(0.7)
        e0_label.shift(UP * 0.3 + RIGHT * 4.0)
        self.play(Create(dashed_line), run_time=1)
        self.play(Create(arrow_to_exit), run_time=1)
        self.play(Create(e0_label), run_time=1)
        self.wait(1)

        # 4. Four additional agents appear with randomized positions
        other_agents = VGroup()
        other_agents_labels = VGroup()
        # Calculate the direction vector from the agent to the exit
        direction = exit_icon.get_center() - agent_circle.get_center()
        direction_norm = direction / np.linalg.norm(direction)

        agent_positions = generate_non_overlapping_positions(
            base_position=agent_circle.get_center(),
            direction_norm=direction_norm,
            min_distance=1.2,
            num_agents=3,
        )
        ll = ["j", "k", "l"]
        for i, pos in enumerate(agent_positions):
            new_agent = Circle(radius=0.5, color=BLUE, fill_opacity=0.5)
            new_agent.move_to(pos)
            other_agents.add(new_agent)
            agent_label = (
                MathTex(rf"{ll[i]}", color=WHITE).scale(0.7).move_to(pos + UP * 0.1)
            )
            # agent_label.shift(UP * 0.8)
            other_agents_labels.add(agent_label)

        self.play(Create(other_agents), run_time=1)
        self.play(Create(other_agents_labels), run_time=1)
        self.wait(1)

        # 5. Dashed arrows from other agents to the first agent
        dashed_lines = VGroup()
        arrow_others = VGroup()
        end_point_others = []
        for agent in other_agents:
            dashed_line = DashedLine(
                start=agent.get_center(),
                end=agent_circle.get_center(),
                color=GREEN,
                stroke_width=2,
            )
            dashed_lines.add(dashed_line)
            direction = agent_circle.get_center() - agent.get_center()
            s = np.linalg.norm(direction)
            direction_norm = direction / s
            length = A * np.exp((1 - s) / D)
            print(f"{length = }, {s = }")
            end_point = agent.get_center() + length * direction_norm
            end_point_others.append(end_point)
            o_arrow = Line(
                start=agent.get_center(),
                end=end_point,
                color=YELLOW,
                stroke_width=2.2,
                buff=0.1,
            )
            o_arrow.add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
            arrow_others.add(o_arrow)
            dashed_lines.add(dashed_line)

        self.play(Create(dashed_lines), run_time=1)
        self.play(Create(arrow_others), run_time=1)
        self.wait(1)

        # 6. Equation appears
        equation = (
            MathTex(
                r"\overrightarrow{e_i} = \frac{1}{N}(\overrightarrow{e_0} + \sum_{j} R(s_{i,j}))"
            )
            .scale(0.8)
            .next_to(agent_circle, UP, buff=2)
        )

        self.play(Write(equation), run_time=2)
        self.wait(1)
        # 7. Lines and equation disappear
        self.play(FadeOut(dashed_lines), FadeOut(equation), run_time=1)
        # 8. Change the direction of the pedestrian
        ####
        e0 = agent_circle.get_center() + direction_to_exit
        vector_sum = (
            e0 - agent_circle.get_center()
        )  # Start with e0 relative to the agent's center
        for end_point in end_point_others:
            vector_sum += end_point - agent_circle.get_center()

        normalized_new_line = vector_sum / np.linalg.norm(vector_sum)
        new_end_point = agent_circle.get_center() + normalized_new_line
        new_line = Line(
            start=agent_circle.get_center(),
            end=new_end_point,
            color=RED,
            stroke_width=2.2,
        )
        new_line.add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        ####
        e_label = MathTex(r"\overrightarrow{e}", color=WHITE).scale(0.7)
        e_label.shift(DOWN * 0.7 + RIGHT * 4.0)
        self.play(ReplacementTransform(arrow_to_exit, new_line), run_time=1)

        self.wait(1)

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

    def setup_visualization_components(self):
        """Set up all visualization components with configurable creation."""
        components = {}

        # Axes setup
        components["axes"] = setup_axes()

        # Velocity equation
        components["velocity_eq"] = (
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

        # Value trackers
        components["T"] = ValueTracker(1)
        components["v0"] = ValueTracker(1)

        # Graph function
        def graph_func(s):
            if s > 6:
                return 0
            current_T = components["T"].get_value()
            current_v0 = components["v0"].get_value()
            if s <= 1:
                return 0
            elif 1 < s <= 1 + current_T * current_v0:
                return min(current_v0, (s - 1) / current_T)
            else:
                return current_v0

        components["graph"] = always_redraw(
            lambda: components["axes"].plot(
                graph_func, x_range=[0, 6], use_smoothing=False, color=BLUE
            )
        )

        # Parameter text and explanation
        components["t_text"] = always_redraw(
            lambda: MathTex(
                rf"T = {components['T'].get_value():.1f}\; [s]",
                font_size=24,
                color=RED if components["T"].get_value() != 1 else WHITE,
            ).next_to(components["velocity_eq"], DOWN, aligned_edge=LEFT, buff=0.8)
        )

        components["v0_text"] = always_redraw(
            lambda: MathTex(
                rf"v_0 = {components['v0'].get_value():.1f}\; [m/s]",
                font_size=24,
                color=RED if components["v0"].get_value() != 1 else WHITE,
            ).next_to(components["velocity_eq"], DOWN, aligned_edge=LEFT, buff=0.8)
        )

        components["explanation_text_T"] = always_redraw(
            lambda: Text(
                get_T_behavior_text(
                    components["T"].get_value(), Tmin=min(T_values), Tmax=max(T_values)
                ),
                font_size=20,
                font="Fira Code Symbol",
                color=YELLOW,
            ).next_to(components["t_text"], DOWN, aligned_edge=LEFT, buff=0.2)
        )

        components["explanation_text_v0"] = always_redraw(
            lambda: Text(
                get_V0_behavior_text(
                    components["v0"].get_value(),
                    vmin=min(v0_values),
                    vmax=max(v0_values),
                ),
                font_size=20,
                font="Fira Code Symbol",
                color=YELLOW,
            ).next_to(components["t_text"], DOWN, aligned_edge=LEFT, buff=0.2)
        )

        # Dashed line and angle visualization
        components["dashed_line"] = always_redraw(
            lambda: DashedLine(
                start=components["axes"].c2p(
                    1 + components["v0"].get_value() * components["T"].get_value(),
                    components["v0"].get_value(),
                ),
                end=components["axes"].c2p(
                    1 + components["v0"].get_value() * components["T"].get_value(),
                    graph_func(1),
                ),
                dash_length=0.1,
                color=BLUE,
            )
        )

        components["filled_angle"] = always_redraw(
            lambda: Sector(
                arc_center=components["axes"].c2p(1, 0),
                inner_radius=0,
                outer_radius=0.5,
                angle=np.arctan(
                    (components["v0"].get_value())
                    / (components["T"].get_value() * components["v0"].get_value())
                ),
                start_angle=0,
                color=BLUE,
                fill_opacity=0.5,
            )
        )

        components["angle"] = always_redraw(
            lambda: Angle(
                Line(
                    start=components["axes"].c2p(1, 0),
                    end=components["axes"].c2p(1 + components["T"].get_value(), 0),
                ),  # Dynamic base line
                Line(
                    start=components["axes"].c2p(1, 0),
                    end=components["axes"].c2p(
                        1 + components["T"].get_value() * components["v0"].get_value(),
                        components["v0"].get_value(),
                    ),
                ),  # Dynamic incline line
                radius=0.5,
                other_angle=False,
                color=ORANGE,
                dot=True,
            )
        )

        return components

    def animate_parameter_changes(self, components, parameter="T"):
        """Generic method to animate parameter changes"""
        values = T_values if parameter == "T" else v0_values
        tracker = components[parameter]

        # Highlight parameter in equation
        if parameter == "T":
            t_framebox = SurroundingRectangle(
                components["velocity_eq"][0][24:25], buff=0.08, color=RED
            )
            t_in_equation = components["velocity_eq"][0][24:25]
            moving_t = MathTex(r"T", font_size=24).move_to(t_in_equation.get_center())

            self.play(FocusOn(t_in_equation))
            self.play(Create(t_framebox))
            self.play(
                t_in_equation.animate.set_color(YELLOW),
                moving_t.animate.move_to(components["t_text"][0][0].get_center()),
                run_time=1,
            )
            self.play(Write(components["t_text"]))
            self.add(components["explanation_text_T"])
        else:
            v0_framebox = SurroundingRectangle(
                components["velocity_eq"][0][38:40], buff=0.15, color=RED
            )
            v0_in_equation = components["velocity_eq"][0][38:40]
            moving_v0 = MathTex(r"v_0", font_size=24).move_to(
                v0_in_equation.get_center()
            )

            self.play(FocusOn(v0_in_equation))
            self.play(Create(v0_framebox))
            self.play(
                v0_in_equation.animate.set_color(YELLOW),
                moving_v0.animate.move_to(components["v0_text"][0][0:2].get_center()),
                run_time=1,
            )
            self.play(Write(components["v0_text"]))
            self.add(components["explanation_text_v0"])
        # Animate parameter changes
        for target_value in values:
            self.play(
                tracker.animate.set_value(target_value).set_color(RED),
                run_time=0.1,
                rate_func=smooth,
            )
            if target_value in (min(values), max(values)):
                self.wait(5)

        self.play(tracker.animate.set_color(WHITE))
        # Return keys of components to fade out
        if parameter == "T":
            self.play(FadeOut(moving_t, t_framebox))
            return ["explanation_text_T", "t_text"]
        else:
            self.play(FadeOut(moving_v0, v0_framebox))
            return ["explanation_text_v0", "v0_text"]

    def construct(self):
        # Modular setup with stage control
        components = self.setup_visualization_components()
        if self.config["agentExit"]:
            self.AgentExitVisualization(A=3.5, D=1)
        if self.config["show_agent_diameter"]:
            self.visualize_agent_diameter(components["axes"])

        if self.config["show_equation"]:
            self.add(components["axes"], components["velocity_eq"])
            self.play(Write(components["velocity_eq"]))
            self.play(Create(components["graph"]))
            self.play(Create(components["dashed_line"]))
            self.play(Create(components["filled_angle"]))
            self.play(Create(components["angle"]))

        if self.config["animate_t_parameter"]:
            fadeouts = self.animate_parameter_changes(components, "T")
            for fo in fadeouts:
                self.play(FadeOut(components[fo]))

        if self.config["animate_v0_parameter"]:
            fadeouts = self.animate_parameter_changes(components, "v0")
            for fo in fadeouts:
                self.play(FadeOut(components[fo]))

        # Fade out scene elements
        self.play(
            FadeOut(components["axes"]),
            FadeOut(components["graph"]),
            FadeOut(components["velocity_eq"]),
            FadeOut(components["v0_text"]),
            FadeOut(components["dashed_line"]),
            FadeOut(components["angle"]),
            FadeOut(components["filled_angle"]),
        )
