"""
Manim animation for visualizing speed function parameters.
Run with: manim -pqh change_v0_T.py
"""

from manim import *
import numpy as np


from dataclasses import dataclass

# Global variable for text font
font = "Inconsolata-dz for Powerline"


# "Fira Code Symbol",
# "Fira Code Symbol",
@dataclass
class VisualizationConfig:
    calculation_s = True
    intro: bool = False
    agent_exit: bool = False
    show_agent_diameter: bool = False
    show_equation: bool = False
    animate_t_parameter: bool = False
    animate_v0_parameter: bool = False


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


def setup_axes(x=10, y=4, xlabel="s [m]", ylabel="v [m/s]"):
    """Set up the axes and add to the scene."""
    axes = Axes(
        x_range=[0, x, 1],
        y_range=[0, y, 1],
        axis_config={"tip_shape": StealthTip},
    )
    x_label = axes.get_x_axis_label(xlabel)
    y_label = axes.get_y_axis_label(ylabel)
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
        base_distance = np.random.uniform(1.5, 3)
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


def calculate_new_direction(arrow_others, e0, agent_circle):
    def redraw():
        end_point_others = [arrow.get_end() for arrow in arrow_others]
        vector_sum = e0 - agent_circle.get_center()
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

        return new_line

    return redraw


def create_agent_direction_rectangle(
    agent_circle, direction_to_exit, length, width=None
):
    """
    Create a rectangle representing the agent's direction with specified parameters

    Parameters:
    - agent_circle: The circle representing the agent
    - direction_to_exit: Normalized direction vector
    - length: Length of the rectangle
    - width: Width of the rectangle (defaults to agent diameter if None)

    Returns:
    - Rectangle object
    - Indices of intersecting agents (if any)
    """
    # Get the center of the agent circle
    center = agent_circle.get_center()

    # Use agent diameter as width if not specified
    if width is None:
        width = agent_circle.radius * 2

    # Create the rectangle
    rectangle = Rectangle(
        height=width,  # height corresponds to width in Manim
        width=length,
        color=BLUE,
        fill_opacity=0.5,
        stroke_width=0,
    ).set_z_index(-1)

    # Rotate the rectangle to align with direction
    angle = np.arctan2(direction_to_exit[1], direction_to_exit[0])
    rectangle.rotate(angle)

    # Position the rectangle starting from agent center
    rectangle.move_to(center + direction_to_exit * (length / 2))

    return rectangle


def find_intersecting_neighbor(
    agent_pos, neighbor_positions, direction, agent_radius=0.5
):
    """
    Find the nearest neighbor intersecting with the agent's movement rectangle.

    Parameters:
    - agent_pos: Position of the primary agent
    - neighbor_positions: List of neighbor positions
    - direction: Normalized direction vector
    - agent_radius: Radius of agents (default 0.5)

    Returns:
    - Index of the nearest intersecting neighbor, or None if no intersection
    """
    # Compute left vector (90-degree rotation)
    left = np.array([-direction[1], direction[0], 0])

    # Stores potential intersecting neighbors
    intersecting_neighbors = []
    print("in find_intersection")
    print(f"{agent_pos = }")
    print(f"{direction = }")
    for idx, neighbor_pos in enumerate(neighbor_positions):
        print(f"{idx = }, {neighbor_pos = }")
        # Vector from agent to neighbor
        dist_p12 = neighbor_pos - agent_pos

        # Check if neighbor is in front (positive dot product)
        in_front = np.dot(direction, dist_p12) >= 0
        if not in_front:
            continue

        # Check if neighbor is within the corridor width
        corridor_width = agent_radius * 2  # Diameter of agent as corridor width
        if abs(np.dot(left, dist_p12)) > corridor_width:
            continue

        # If we reach here, the neighbor intersects the rectangle
        intersecting_neighbors.append((idx, np.linalg.norm(dist_p12)))

    print("==========")
    print(f"{corridor_width = }")
    print(f"{neighbor_positions = }")
    print(f"{intersecting_neighbors = }")

    print("==========")
    # Return the index of the nearest intersecting neighbor
    if intersecting_neighbors:
        ret = min(intersecting_neighbors, key=lambda x: x[1])[0]
        print(">>>>> Return index ", ret)
        return min(intersecting_neighbors, key=lambda x: x[1])[0]

    return None


class ChangingTAndV0(Scene):
    def __init__(
        self, *args, config: VisualizationConfig = VisualizationConfig(), **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config

    def dynamic_neighbor_highlight(
        self, agent_circle, other_agents, exit_icon, neighbor_positions
    ):
        """
        Dynamically highlights a neighboring agent in the direction of an exit.

        Parameters:
        agent_circle: The representation of the agent (e.g., a graphical object).
        exit_icon: Graphical representation of the exit.
        neighbor_positions: List of positions of neighboring agents.

        Returns:
        None
        """

        def update_highlight():
            print("Update Highlight Called")
            print(f"{exit_icon.get_center() = }")
            # Calculate direction based on current exit position
            direction = exit_icon.get_center() - agent_circle.get_center()
            direction_norm = direction / np.linalg.norm(direction)
            print(f"in update_highlight")
            print(f"Direction vector: {direction}, Normalized: {direction_norm}")
            # Find intersecting neighbor
            intersecting_idx = find_intersecting_neighbor(
                agent_circle.get_center(), neighbor_positions, direction_norm
            )
            for other_agent in other_agents:
                # # Remove previous highlight if exists
                other_agent.set_color(BLUE)

            if intersecting_idx is not None:
                print(f"Hightlight Agent with index {intersecting_idx}")
                print(f"Dot position: {neighbor_positions[intersecting_idx]}")
                other_agents[intersecting_idx].set_color(RED)

            return intersecting_idx

        return update_highlight()
        # self.add_updater(lambda _: update_highlight())

    def animate_calculation_s(self):
        # Agent circle and label
        agent_circle = Circle(radius=0.5, color=BLUE, fill_opacity=0.5)
        agent_label = (
            MathTex(r"i", color=WHITE)
            .scale(0.7)
            .move_to(agent_circle.get_center() + LEFT * 0.1)
        )
        self.play(GrowFromCenter(agent_circle), Create(agent_label))

        # Create trackers for exit position
        exit_tracker = ValueTracker(0.5)

        # Always redrawn exit icon
        exit_icon = always_redraw(
            lambda: ImageMobject("exit.png")
            .scale(0.1)
            .next_to(agent_circle, RIGHT + UP * exit_tracker.get_value(), buff=4.5)
        )

        # Always redrawn direction calculations
        def get_direction():
            return exit_icon.get_center() - agent_circle.get_center()

        def get_normalized_direction():
            direction = get_direction()
            return direction / np.linalg.norm(direction)

        # Always redrawn dashed line
        dashed_line = always_redraw(
            lambda: DashedLine(
                start=agent_circle.get_center(),
                end=exit_icon.get_center(),
                color=GREEN,
                stroke_width=2,
            )
        )

        # Always redrawn direction arrow
        arrow_to_exit = always_redraw(
            lambda: Line(
                start=agent_circle.get_center(),
                end=agent_circle.get_center() + get_normalized_direction(),
                color=YELLOW,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        )

        # Always redrawn e0 label
        e0_label = always_redraw(
            lambda: MathTex(r"\overrightarrow{e_0}", color=WHITE)
            .scale(0.7)
            .next_to(arrow_to_exit, buff=0.0)
            .shift(DOWN * 0.25 + LEFT * 0.3)
        )

        # Add all elements
        self.play(FadeIn(exit_icon))
        self.add(dashed_line, arrow_to_exit, e0_label)
        agent_positions = (
            [agent_circle.get_center() + RIGHT * 2 + UP * 0.7]
            + [agent_circle.get_center() + RIGHT * 3 + DOWN * 0.6]
            + [agent_circle.get_center() + UP * 1.2 + RIGHT * 0.6]
            + [agent_circle.get_center() + DOWN * 1.2 + RIGHT * 0.6]
            + [agent_circle.get_center() + LEFT * 1.5 + UP * 0.5]
            + [agent_circle.get_center() + LEFT * 2 + DOWN * 0.7]
            + [agent_circle.get_center() + LEFT * 3 + UP * 0.2]
        )
        print(agent_positions)
        other_agents = VGroup()
        for i, pos in enumerate(agent_positions):
            new_agent = Circle(radius=0.5, color=BLUE, fill_opacity=0.5)
            new_agent.move_to(pos)
            other_agents.add(new_agent)

        self.play(Create(other_agents), run_time=1)
        self.wait(2)
        # Always redrawn rectangle
        rectangle = always_redraw(
            lambda: create_agent_direction_rectangle(
                agent_circle, get_normalized_direction(), length=2.0
            ).set_z_index(-1)
        )
        self.add(rectangle)
        self.wait(1)
        index = self.dynamic_neighbor_highlight(
            agent_circle, other_agents, exit_icon, agent_positions
        )
        if index:
            other_agents[index].set_color(RED)

        # Animate exit position changes
        # for new_offset in [0, 0.5, -0.5, 0]:
        #     self.play(
        #         exit_tracker.animate.set_value(new_offset),
        #         run_time=2,
        #     )

        self.wait(2)

    def animate_equation(
        self,
        initial_equation,
        final_equation,
        font_size=32,
    ):
        """
        Animate the transformation of an equation with optional brace and label.

        Parameters:
        - initial_equation: The starting Tex or MathTex object
        - final_equation: The final MathTex object to transform into
        - brace_part: The part of the equation to apply the brace to (optional)
        - brace_label: The label for the brace (optional)
        - font_size: Base font size for the animation
        """
        # Preserve original positioning
        original_position = initial_equation.get_center()
        t1 = Text(
            r"Initially, each agent follows its own desired path.",
            font_size=font_size - 4,
            # tex_template=TexFontTemplates.french_cursive,
            font=font,
        ).next_to(original_position, UP * 2)
        t2 = Text(
            r"Now, agents' directions are collectively shaped by interactions.",
            font_size=font_size - 4,
            # tex_template=TexFontTemplates.french_cursive,
            font=font,
        ).next_to(original_position, UP * 2)
        t3 = Text(
            r"Normalizing agent's direction.",
            font_size=font_size - 4,
            # tex_template=TexFontTemplates.french_cursive,
            font=font,
        ).next_to(original_position, UP * 2)

        self.play(FadeIn(initial_equation[0]), run_time=0.5)
        self.play(FadeIn(t1), run_time=0.1)

        self.wait(2)
        self.play(FadeIn(initial_equation[1]), run_time=0.5)
        # self.play(t1.animate.set_opacity(0), t2.animate.set_opacity(1), run_time=1)
        self.play(Transform(t1, t2))
        brace_part = initial_equation[1][1:]
        underbracket = Brace(brace_part, color=YELLOW)
        # underbracket.move_to(brace_part.get_center())
        underbracket_label = MathTex(
            r"\overrightarrow{e_{ij}}", font_size=font_size
        ).next_to(underbracket, DOWN)
        # Animate brace and label
        self.play(
            FadeIn(underbracket, run_time=0.5),
            FadeIn(underbracket_label, shift=DOWN, run_time=1),
        )
        self.wait(2)

        # Prepare 1/N fraction for transformation
        n_fraction = (
            MathTex(r"\frac{1}{N}", font_size=font_size + 4)
            .scale(0.8)
            .next_to(initial_equation)
            .shift(RIGHT)
        )

        self.play(
            FadeOut(initial_equation, run_time=0.1),
            # Fade out original elements
            # Transform 1/N to final equation
            Transform(t1, t3),
            TransformMatchingShapes(n_fraction, final_equation, run_time=1),
        )
        self.wait(2)
        # self.play(FadeOut(t3))
        return final_equation, underbracket, underbracket_label, t1

    def ShowIntro(self):
        title = Text("The collision-free speed model", font_size=50).shift(UP * 2)
        text = MarkupText(
            """
            The collision-free speed model<sup>1</sup> is a mathematical approach
            designed for pedestrian dynamics, emphasizing the prevention
            of collisions among agents.
            """,
            font=font,
            font_size=24,
        )
        text2 = Text(
            r"""
            The model describes pedestrian movement through
            a first-order ordinary differential equation
            that governs the velocity of each pedestrian.
            
            Mathematically, this is expressed as a derivative
            equation representing the instantaneous rate of change
            of a agents's velocity over time.

            """,
            font=font,
            font_size=24,
        )
        eq = (
            MathTex(
                r"\dot{x}_i = ",
                r"V_i(s_i)",
                r"\times",
                r"e_i(x_i, x_j, \cdots)",
                font_size=40,
            )
            .set_color_by_tex_to_color_map(
                {
                    "V_i": RED,  # Set the speed function in red
                    "e_i": BLUE,  # Set the direction function in blue
                    "\\dot{x}_i": YELLOW,
                }
            )
            .next_to(text2, DOWN * 1.5)
        )
        text3 = Text(
            """
            The speed function regulates the overall speed of the agent,
            """,
            font=font,
            font_size=24,
            t2c={"speed function": RED, "overall speed": YELLOW},
        )
        text4 = Text(
            """
            while the direction function determines the direction
            in which the agent moves.
            """,
            font=font,
            font_size=24,
            t2c={"direction function": BLUE},
        )

        ref = MarkupText(
            """
             <sup>1</sup> Tordeux, A., Chraibi, M., Seyfried, A. (2016).
            Collision-Free Speed Model for Pedestrian Dynamics.
            In: Knoop, V., Daamen, W. (eds) Traffic and Granular Flow ‘15.
            https://doi.org/10.1007/978-3-319-33482-0_29
            """,
            font_size=18,
            font=font,
            color="Gray",
        ).next_to(text, DOWN, buff=2)
        speed_index = 3
        direction_index = 10
        self.play(FadeIn(title))
        self.play(FadeIn(text))
        self.play(FadeIn(ref))
        self.wait(4)
        self.play(FadeOut(ref))

        self.wait(2)
        self.play(Transform(text, text2), FadeIn(eq))
        self.wait(2)
        self.play(Transform(text, text3))
        self.play(
            Circumscribe(
                eq[1],
                color=RED,
            ),
            Wiggle(text3[speed_index:16], color=WHITE),
            run_time=2,
        )

        self.wait(2)
        self.play(Transform(text, text4), FadeOut(text3))
        self.play(
            Circumscribe(
                eq[3],
                color=BLUE,
            ),
            Wiggle(text4[direction_index:25]),
            run_time=2,
        )
        self.wait(3)
        self.play(FadeOut(*self.mobjects))
        self.wait(3)

    def AgentExitVisualization(self, components):
        A_tracker = components["A"]
        D_tracker = components["D"]
        fs = 32  # font size
        # 1. Agent appears
        agent_circle = Circle(radius=0.5, color=BLUE, fill_opacity=0.5)
        agent_label = (
            MathTex(r"i", color=WHITE)
            .scale(0.7)
            .move_to(agent_circle.get_center() + LEFT * 0.1)
        )
        text = Text("The direction function", font=font, font_size=40)
        difference = Text(
            """
            The direction function defines the direction in which an agent moves.
            It is computed as a superposition of the desired direction towards a goal
            and the influence of neighboring agents.
            """,
            font_size=20,
            font=font,
            t2c={
                "direction function": BLUE,
                "desired direction": ORANGE,
                "neighboring agents": ORANGE,
            },
        )
        intro_text = (
            VGroup(text, difference)
            .arrange(DOWN, buff=1)
            .next_to(agent_circle, UP, buff=1)
        )
        self.play(FadeIn(intro_text))
        self.wait(5)
        self.play(FadeOut(intro_text))
        self.play(GrowFromCenter(agent_circle), Create(agent_label))
        # 2. Exit icon appears
        exit_icon = ImageMobject("exit.png").scale(0.5)
        exit_icon.scale(0.3).next_to(agent_circle, RIGHT, buff=4.5)
        self.play(FadeIn(exit_icon), run_time=1)
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
        e0 = agent_circle.get_center() + direction_to_exit
        self.play(Create(dashed_line), run_time=1)
        self.play(Create(arrow_to_exit), run_time=1)
        e0_label.next_to(arrow_to_exit, buff=0.0).shift(DOWN * 0.25 + LEFT * 0.3)
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
        ll = ["n", "k", "m"]
        for i, pos in enumerate(agent_positions):
            new_agent = Circle(radius=0.5, color=BLUE, fill_opacity=0.5)
            new_agent.move_to(pos)
            other_agents.add(new_agent)
            _agent_label = (
                MathTex(rf"{ll[i%3]}", color=WHITE).scale(0.7).move_to(pos + UP * 0.1)
            )
            other_agents_labels.add(_agent_label)

        self.play(Create(other_agents), run_time=1)
        self.play(Create(other_agents_labels), run_time=1)
        self.wait(1)
        equation = (
            Tex(
                r"$\overrightarrow{e_i} = \overrightarrow{e_0}$",
                r"$+ \sum_{j} A\exp(\frac{l-s}{D})$",
                font_size=fs + 4,
            )
            .scale(0.8)
            .next_to(agent_circle, UP, buff=2)
        )
        _final_equation = (
            MathTex(
                r"\overrightarrow{e_i} = \frac{1}{N}\Big(\overrightarrow{e_0} + \sum_{j} A\exp(\frac{l-s}{D})\Big)",
                font_size=fs + 4,
            )
            .scale(0.8)
            .move_to(equation)
        )

        final_equation, underbracket, underbracket_label, text3 = self.animate_equation(
            equation,
            _final_equation,
            font_size=fs,
        )

        A_in_equation = final_equation[0][16:17]
        D_in_equation = final_equation[0][25:26]
        moving_A = MathTex(r"A", font_size=fs).move_to(A_in_equation.get_center())
        moving_D = MathTex(r"D", font_size=fs).move_to(D_in_equation.get_center())

        self.play(FocusOn(A_in_equation))
        self.play(
            moving_A.animate.next_to(equation, RIGHT, buff=2),
            run_time=2,
        )
        self.play(FocusOn(D_in_equation))
        self.play(
            moving_D.animate.next_to(moving_A, DOWN),
            run_time=2,
        )
        # 5. Dashed arrows from other agents to the first agent
        dashed_lines = VGroup()
        arrow_others = VGroup()
        end_point_others = []
        value_text_A = always_redraw(
            lambda: MathTex(
                f"={components['A'].get_value():.1f}",
                color=WHITE,
                font_size=fs,
            ).next_to(moving_A)
        )
        self.add(value_text_A)
        value_text_D = always_redraw(
            lambda: MathTex(
                f"={components['D'].get_value():.1f}",
                color=WHITE,
                font_size=fs,
            ).next_to(moving_D)
        )
        self.add(value_text_D)
        group = VGroup(moving_A, moving_D, value_text_A, value_text_D)
        rectangle = SurroundingRectangle(group, color=YELLOW, buff=0.2)
        rect_label = Text(
            r"Direction parameters",
            font_size=fs - 4,
            # tex_template=TexFontTemplates.french_cursive,
            font=font,
        ).next_to(rectangle, UP)

        self.play(Create(rectangle), Transform(text3, rect_label))

        arrow_others = always_redraw(
            lambda: VGroup(
                *[
                    Line(
                        start=agent.get_center(),
                        end=agent.get_center()
                        + A_tracker.get_value()
                        * np.exp(
                            (
                                1
                                - np.linalg.norm(
                                    agent_circle.get_center() - agent.get_center()
                                )
                            )
                            / D_tracker.get_value()
                        )
                        * (agent_circle.get_center() - agent.get_center())
                        / np.linalg.norm(
                            agent_circle.get_center() - agent.get_center()
                        ),
                        color=YELLOW,
                        stroke_width=2.2,
                        buff=0.1,
                    ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
                    for agent in other_agents
                ]
            )
        )

        for agent in other_agents:
            _dashed_line = DashedLine(
                start=agent.get_center(),
                end=agent_circle.get_center(),
                color=GREEN,
                stroke_width=2,
            )
            dashed_lines.add(_dashed_line)
            direction = agent_circle.get_center() - agent.get_center()
            s = np.linalg.norm(direction)
            direction_norm = direction / s
            length = A_tracker.get_value() * np.exp((1 - s) / D_tracker.get_value())
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

        # 7. Lines and equation disappear
        self.play(FadeOut(dashed_lines), run_time=1)
        # 8. Change the direction of the pedestrian
        ####
        # Dynamically update the pedestrian's direction
        new_line = always_redraw(
            calculate_new_direction(arrow_others, e0, agent_circle)
        )
        e_label = always_redraw(
            lambda: MathTex(r"\overrightarrow{e_i}", color=RED)
            .scale(0.7)
            .next_to(new_line, buff=0.1)
            .shift(UP * 0.2)
        )

        self.add(new_line)
        self.add(e_label)
        # ####
        #        self.play(ReplacementTransform(arrow_to_exit, new_line), run_time=1)
        # Animate changes in A and D to see dynamic updates
        axes = (
            setup_axes(x=4, y=8, xlabel="", ylabel="")
            .scale(0.4)
            .next_to(agent_circle, LEFT)
        )
        xlabel = MathTex("s [m]", font_size=30).next_to(axes, DOWN)
        ylabel = MathTex(r"\overrightarrow{e_{ij}}", font_size=36).next_to(axes, LEFT)

        self.add(xlabel, ylabel)

        def exp_func(s):
            return A_tracker.get_value() * np.exp((-s) / D_tracker.get_value())

        dir_graph = always_redraw(
            lambda: axes.plot(exp_func, x_range=[0, 3], use_smoothing=False, color=BLUE)
        )
        self.play(Create(axes))
        self.play(Create(dir_graph))
        new_D = 2
        D_tracker.set_value(new_D)
        for new_A in [3, 5.5, 3]:
            self.play(
                Circumscribe(moving_A, color=RED, time_width=0.1),
                A_tracker.animate.set_value(new_A),
                run_time=2,
            )
        new_A = 3
        A_tracker.set_value(new_A)
        for new_D in [2, 0.5, 2]:
            self.play(
                Circumscribe(moving_D, color=RED, time_width=0.1),
                D_tracker.animate.set_value(new_D),
                run_time=2,
            )
        value_text_D.animate.set_color(WHITE).scale(1)
        ###############################################
        self.play(
            FadeOut(
                dashed_line,
                _final_equation,
                agent_circle,
                other_agents,
                other_agents_labels,
                exit_icon,
                rect_label,
                text3,
                group,
                underbracket,
                underbracket_label,
                agent_label,
                dashed_lines,
                new_line,
                axes,
                dir_graph,
                xlabel,
                ylabel,
                e_label,
                e0_label,
                rectangle,
                value_text_A,
                value_text_D,
                arrow_to_exit,
                arrow_others,
            ),
        )
        self.wait(5)

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
        if self.config.calculation_s:
            self.animate_calculation_s()
        if self.config.agent_exit:
            # Value trackers
            components["A"] = ValueTracker(3)
            components["D"] = ValueTracker(2)

        if self.config.show_equation:
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
                        components["T"].get_value(),
                        Tmin=min(T_values),
                        Tmax=max(T_values),
                    ),
                    font_size=20,
                    font=font,
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
                    font=font,
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
                            1
                            + components["T"].get_value()
                            * components["v0"].get_value(),
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
        if self.config.intro:
            self.ShowIntro()
        if self.config.agent_exit:
            self.AgentExitVisualization(components)
        if self.config.show_agent_diameter:
            self.visualize_agent_diameter(components["axes"])

        if self.config.show_equation:
            self.add(components["axes"], components["velocity_eq"])
            self.play(Write(components["velocity_eq"]))
            self.play(Create(components["graph"]))
            self.play(Create(components["dashed_line"]))
            self.play(Create(components["filled_angle"]))
            self.play(Create(components["angle"]))

        if self.config.animate_t_parameter:
            fadeouts = self.animate_parameter_changes(components, "T")
            for fo in fadeouts:
                self.play(FadeOut(components[fo]))

        if self.config.animate_v0_parameter:
            fadeouts = self.animate_parameter_changes(components, "v0")
            for fo in fadeouts:
                self.play(FadeOut(components[fo]))

        # Fade out scene elements
        if any(
            [
                self.config.show_equation,
                self.config.animate_t_parameter,
                self.config.animate_v0_parameter,
            ]
        ):
            self.play(
                FadeOut(components["axes"]),
                FadeOut(components["graph"]),
                FadeOut(components["velocity_eq"]),
                FadeOut(components["v0_text"]),
                FadeOut(components["dashed_line"]),
                FadeOut(components["angle"]),
                FadeOut(components["filled_angle"]),
            )
