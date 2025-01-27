from manim import *
import numpy as np
import manim as m


# Global variable for text font
font = "Inconsolata-dz for Powerline"
font_size_text = 30


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


def compute_total_influence(target_agent, other_agents):
    total_influence = np.zeros(3)
    for other_agent in other_agents:
        influence, _, dist, strength = neighbor_repulsion(target_agent, other_agent)
        total_influence += influence
    return total_influence


def calculate_new_direction(
    direction,
    agent_position,
    agent_radius,
    wall,
    wall_buffer_distance,
    pushout_strength=0.1,
):
    """
    Simulates the HandleWallAvoidance function for visualization.

    Parameters:
    - direction: Current movement direction of the agent (numpy array).
    - agent_position: Position of the agent (numpy array).
    - agent_radius: Radius of the agent.
    - wall: A tuple of two points representing the wall ((p1, p2)).
    - wall_buffer_distance: Buffer distance from the wall.
    - pushout_strength: Strength of pushout force near the wall.

    Returns:
    - New normalized direction as a numpy array.
    """
    critical_wall_distance = wall_buffer_distance + agent_radius
    influence_start_distance = 2.0 * critical_wall_distance

    # Define wall points
    p1, p2 = np.array(wall[0]), np.array(wall[1])

    # Calculate the closest point on the wall
    wall_vector = p2 - p1
    wall_length = np.linalg.norm(wall_vector)
    wall_direction = wall_vector / wall_length

    relative_position = agent_position[:2] - p1[:2]
    projection_length = np.dot(relative_position, wall_direction[:2])

    if projection_length < 0:
        closest_point = p1
    elif projection_length > wall_length:
        closest_point = p2
    else:
        closest_point = p1 + projection_length * wall_direction

    # Calculate distance to the wall
    distance_vector = agent_position[:2] - closest_point[:2]
    perpendicular_distance = np.linalg.norm(distance_vector)
    if perpendicular_distance == 0:
        direction_away_from_boundary = np.array([0, 0])
    else:
        direction_away_from_boundary = distance_vector / perpendicular_distance

    if perpendicular_distance < critical_wall_distance:
        parallel_component = wall_direction[:2] * np.dot(
            direction[:2], wall_direction[:2]
        )
        new_direction = (
            parallel_component + direction_away_from_boundary * pushout_strength
        )
        return np.array(
            [*new_direction[:2] / np.linalg.norm(new_direction), 0]
        ), "critical"
    elif perpendicular_distance < influence_start_distance:
        dot_product = np.dot(direction[:2], direction_away_from_boundary)
        if dot_product < 0:
            influence_factor = (influence_start_distance - perpendicular_distance) / (
                influence_start_distance - critical_wall_distance
            )
            parallel_component = wall_direction[:2] * np.dot(
                direction[:2], wall_direction[:2]
            )
            perpendicular_component = direction[:2] - parallel_component
            new_direction = parallel_component + perpendicular_component * (
                1.0 - influence_factor
            )
            return np.array(
                [*new_direction[:2] / np.linalg.norm(new_direction), 0]
            ), "influence"

    return np.array([*direction[:2] / np.linalg.norm(direction), 0]), "none"


def to_3d(vec):
    """
    Converts a 2D vector to a 3D vector by appending a 0 for the z-component.
    """
    return np.array([vec[0], vec[1], 0])


def neighbor_repulsion(agent1, agent2):
    """
    Calculate the repulsion effect between two agents based on the Anticipation Velocity Model.

    Parameters:
    - agent1, agent2: Dictionaries representing agents, including their positions, orientations,
      velocities, radii, and model-specific parameters.

    Returns:
    - A numpy array representing the influence direction scaled by interaction strength.
    """
    # Unpack agent properties
    pos1, pos2 = agent1["position"], agent2["position"]
    radius1, radius2 = agent1["radius"], agent2["radius"]
    velocity1, velocity2 = agent1["velocity"], agent2["velocity"]
    anticipation_time = agent1["anticipation_time"]
    orientation1, orientation2 = agent1["orientation"], agent2["orientation"]
    destination1 = agent1["destination"]

    # Compute distance vector and adjusted distance
    dist_vector = pos2[:2] - pos1[:2]  # Work in 2D for calculations
    distance = np.linalg.norm(dist_vector)
    ep12 = dist_vector / distance if distance > 0 else np.zeros(2)
    adjusted_dist = distance - (radius1 + radius2)

    # Compute movement and desired directions
    d1 = (destination1[:2] - pos1[:2]) / np.linalg.norm(destination1[:2] - pos1[:2])
    d2 = (
        velocity2[:2] / np.linalg.norm(velocity2[:2])
        if np.linalg.norm(velocity2[:2]) > 0
        else np.zeros(2)
    )

    # Check perception range (Eq. 1)
    in_perception_range = (np.dot(d1, ep12) >= 0) or (
        np.dot(orientation1[:2], ep12) >= 0
    )

    # Compute S_Gap and R_dist
    s_gap = np.dot(velocity1[:2] - velocity2[:2], ep12) * anticipation_time
    r_dist = max(adjusted_dist - s_gap, 0)
    if not in_perception_range:
        return (
            np.zeros(3),
            s_gap,
            r_dist,
            0,
        )  # Return a 3D zero vector for consistency

    # Interaction strength (Eq. 3 & 4)
    alignment_base = 1.0
    alignment_weight = 0.5
    alignment_factor = alignment_base + alignment_weight * (
        1.0 - np.dot(d1, orientation2[:2])
    )
    interaction_strength = (
        agent1["strength"] * alignment_factor * np.exp(-r_dist / agent1["range"])
    )

    # Compute adjusted influence direction
    newep12 = dist_vector + velocity2[:2] * anticipation_time
    influence_direction = calculate_influence_direction(d1, newep12)

    # Return as 3D vector for consistency with Manim
    return (
        to_3d(influence_direction * interaction_strength),
        s_gap,
        r_dist,
        interaction_strength,
    )


def calculate_influence_direction(d1, newep12, random_seed=42):
    """
    Calculate the adjusted influence direction based on desired and predicted directions.
    """
    np.random.seed(random_seed)
    if np.linalg.norm(newep12) > 0:
        orthogonal_direction = np.array([-d1[1], d1[0]]) / np.linalg.norm(d1)
        alignment = np.dot(orthogonal_direction, newep12 / np.linalg.norm(newep12))

        if abs(alignment) < 1e-6:  # J_EPS equivalent
            # Randomly choose left or right if alignment is close to zero
            if np.random.randint(2) == 0:
                return -orthogonal_direction
        elif alignment > 0:
            return -orthogonal_direction
        return orthogonal_direction

    return np.zeros(2)


def project_point(point, line_start, line_vector):
    # Vector from line start to point
    point_vector = point - line_start

    # Projection of point onto line
    projection_scalar = np.dot(point_vector, line_vector) / np.dot(
        line_vector, line_vector
    )
    projection = line_start + projection_scalar * line_vector

    return projection


def update_agent_state(agent, resulting_direction, exit_position, dt=0.1):
    # Update position
    resulting_direction = np.array(resulting_direction, dtype=float)
    agent["position"] = np.array(agent["position"], dtype=float)
    agent["position"] += resulting_direction * dt

    # Update velocity
    agent["velocity"] = resulting_direction * 0.5

    # Update orientation
    agent["orientation"] = resulting_direction[:2] / np.linalg.norm(
        resulting_direction[:2]
    )
    agent["desired_direction"] = (exit_position - agent["position"]) / np.linalg.norm(
        exit_position - agent["position"]
    )

    return agent


class NeighborInteraction(Scene):
    def ShowIntro(self):
        title = Text(
            "The Anticipation velocity model", font_size=font_size_text + 5, font=font
        ).align_on_border(UP)
        text = MarkupText(
            """
            The Anticipation Velocity Model (AVM)<sup>1</sup>
            is a mathematical approach designed for
            pedestrian dynamics.
            """,
            font=font,
            font_size=font_size_text,
        )
        text01 = MarkupText(
            """
            The model is based on the Collision-Free Speed Model.
            Key distinctions include:

            - Influence direction: Orthogonal to desired movement,
              rather than directly towards agents
            - Influence distance: Proactively anticipated,
              enabling more sophisticated collision prediction
            """,
            font=font,
            font_size=font_size_text,
        )

        text2 = Text(
            r"""
            The model describes pedestrian movement through
            a first-order ordinary differential equation
            that governs the velocity of each pedestrian.

            Mathematically, this is expressed as a derivative
            equation representing the instantaneous rate of change
            of a agent's velocity over time.

            """,
            font=font,
            font_size=font_size_text,
        )
        eq = (
            MathTex(
                r"\overrightarrow{\dot{x}}_i",
                r"=",
                r"V_i(s_i)",
                r"\times",
                r"\overrightarrow{e_i}(x_i, x_j, \cdots)",
                font_size=50,
            )
            .set_color_by_tex_to_color_map(
                {
                    r"V_i": RED,  # Set the speed function in red
                    r"\overrightarrow{e_i}": BLUE,  # Set the direction function in blue
                    r"\overrightarrow{\dot{x}}_i": YELLOW,
                }
            )
            .next_to(text2, DOWN * 1.5)
        )
        text3 = Text(
            """
            The speed function regulates the overall speed
            of the agent
            """,
            font=font,
            font_size=font_size_text,
            t2c={"speed function": RED, "overall speed": YELLOW},
        )
        text4 = Text(
            """
            while the direction function determines the direction
            in which the agent moves.
            """,
            font=font,
            font_size=font_size_text,
            t2c={"direction function": BLUE},
        )
        text5 = Text(
            "Video Overview:\n"
            "- Anticipated distance calculation\n"
            "- Neighbor's influence on the direction\n"
            "- Wall's influence on agents\n"
            "- Simulations for model demonstration\n",
            font=font,
            font_size=font_size_text,
            line_spacing=1.5,
        )

        ref = MarkupText(
            """
             <sup>1</sup> Xu, Q., Chraibi, M., Seyfried, A. (2021).
            Anticipation in a velocity-based model for pedestrian dynamics.
            Transportation Research Part C: Emerging Technologies
            10.1016/j.trc.2021.103464
            """,
            font_size=18,
            font=font,
            color="Gray",
        ).to_corner(DOWN + LEFT)  # next_to(text, DOWN, buff=1)
        speed_index = 3
        direction_index = 10
        self.play(FadeIn(title))
        self.play(FadeIn(text))
        self.play(FadeIn(ref))
        self.wait(3)
        self.play(FadeOut(ref))
        self.wait(1)
        self.play(Transform(text, text01))
        self.wait(10)
        self.play(Transform(text, text2), FadeIn(eq))
        self.wait(8)
        self.play(Transform(text, text3))
        self.wait(3)
        overall_speed_position = text3[34:36].get_center()
        eq_position = eq[0].get_left()
        # Create an arrow going from "overall speed" to eq[0]
        # arrow = Arrow(
        #     start=overall_speed_position, end=eq_position, color=YELLOW, buff=1
        # )
        self.play(
            Circumscribe(
                eq[2],
                color=RED,
            ),
            Indicate(text3[speed_index:16], color=RED),
            run_time=2,
        )
        self.wait(1)
        self.play(Transform(text, text4), FadeOut(text3))
        self.play(
            Circumscribe(
                eq[4],
                color=BLUE,
            ),
            Indicate(text4[direction_index:25], color=BLUE),
            run_time=2,
        )

        self.wait(2)
        self.play(FadeOut(*self.mobjects))
        self.wait(1)
        self.play(Write(text5))
        self.wait(2)
        self.play(FadeOut(text5))

    def create_predicted_distance_act(
        self,
        pos1=np.array([-2, -2, 0]),
        velocity1=np.array([1, 1, 0]),
        pos2=np.array([2, -2, 0]),
        velocity2=np.array([-1, 2, 0]),
        anticipation_time=1.0,
        radius=0.4,
        velocity_scale=0.5,
    ):
        agent_label = Text(
            r"""
            Agents are modeled as constant circles.

            While they may have different sizes,
            their circular shapes remain unchanged.
            """,
            font_size=font_size_text,
            font=font,
        ).align_on_border(UP)

        self.add(agent_label)
        self.wait(4)
        circle_radius = 1  # Assuming l = 1 diameter
        circle = Circle(radius=circle_radius, color=BLUE, fill_opacity=0.3)
        semicircle = Arc(radius=1, angle=PI, color=GREEN, fill_opacity=0.3)

        dot = Dot()
        # self.add(dot)
        self.play(GrowFromCenter(circle))

        dot2 = dot.copy().shift(RIGHT).set_color(BLUE)
        dot3 = dot2.copy().set_color(BLUE)

        self.play(Transform(dot, dot2))
        self.play(MoveAlongPath(dot2, semicircle), run_time=1, rate_func=linear)
        line = Line(dot3, dot2)
        diameter_label = MathTex(r"l", font_size=28).next_to(line, DOWN)
        self.add(line)
        self.play(Write(diameter_label))
        self.wait(2)
        self.play(
            FadeOut(agent_label),
            FadeOut(diameter_label),
            FadeOut(line),
            FadeOut(dot),
            FadeOut(dot2),
            FadeOut(dot3),
        )
        title = Text(
            "Predicted Distance with a time constant.",
            font_size=font_size_text,
            font=font,
        ).to_edge(UP)
        self.play(Write(title))

        # Draw agents
        agent1_circle = Circle(radius=radius, color=BLUE, fill_opacity=0.5).move_to(
            pos1
        )
        agent2_circle = Circle(radius=radius, color=RED, fill_opacity=0.5).move_to(pos2)
        # Draw velocity vectors
        velocity1_arrow = Arrow(
            start=pos1, end=pos1 + velocity1 * velocity_scale, color=GREEN
        )
        velocity2_arrow = Arrow(
            start=pos2, end=pos2 + velocity2 * velocity_scale, color=GREEN
        )
        # Anticipated positions
        anticipated_pos1 = pos1 + velocity1 * anticipation_time
        anticipated_pos2 = pos2 + velocity2 * anticipation_time

        # Dashed lines to anticipated positions
        dashed_line1 = DashedLine(
            start=pos1,
            end=anticipated_pos1,
            color=BLUE,
            dash_length=0.1,
            stroke_width=2,
        )
        dashed_line2 = DashedLine(
            start=pos2,
            end=anticipated_pos2,
            color=RED,
            dash_length=0.1,
            stroke_width=2,
        )
        anticipated_circle1 = Circle(
            radius=radius,
            color=BLUE,
            fill_opacity=0.4,
            stroke_width=1,
        ).move_to(pos1)

        anticipated_circle2 = Circle(
            radius=radius,
            color=RED,
            fill_opacity=0.4,
            stroke_width=1,
        ).move_to(pos2)
        # Interaction distance (s_ij^a)
        interaction_line = DashedLine(
            start=pos1, end=pos2, color=WHITE, dash_length=0.1, stroke_width=2
        )
        interaction_line_anticipated = DashedLine(
            start=anticipated_pos1,
            end=anticipated_pos2,
            color=WHITE,
            dash_length=0.1,
            stroke_width=2,
        )
        # Projections
        interaction_vector = pos2 - pos1
        interaction_length = np.linalg.norm(interaction_vector)
        proj1 = project_point(anticipated_pos1, pos1, interaction_vector)
        proj2 = project_point(anticipated_pos2, pos1, interaction_vector)
        s_line = Line(start=proj1, end=proj2, color=YELLOW)
        interaction_label = MathTex("s", font_size=40).move_to(
            s_line.get_center() + DOWN * 0.5
        )
        interaction_label1 = interaction_label.copy()
        interaction_label2 = interaction_label.copy()
        # Dashed lines to projections
        proj_interaction_line1 = DashedLine(
            start=anticipated_pos1,
            end=proj1,
            color=WHITE,
            dash_length=0.1,
            stroke_width=2,
        )
        proj_interaction_line2 = DashedLine(
            start=anticipated_pos2,
            end=proj2,
            color=WHITE,
            dash_length=0.1,
            stroke_width=2,
        )

        # Placement of objects --- Here starts the choreography
        waiting_time = 2
        self.play(Transform(circle, agent1_circle))
        self.play(GrowFromCenter(agent2_circle))
        self.play(Create(interaction_line))
        self.wait(waiting_time)
        # Animations to move circles from original to anticipated positions
        self.play(
            Create(dashed_line1),
            Create(dashed_line2),
            anticipated_circle1.animate.move_to(anticipated_pos1),
            anticipated_circle2.animate.move_to(anticipated_pos2),
        )
        self.wait(waiting_time)
        self.play(Create(interaction_line_anticipated))
        self.wait(waiting_time)
        self.play(Create(proj_interaction_line1), Create(proj_interaction_line2))
        self.play(FadeIn(s_line), FadeIn(interaction_label))
        self.wait(waiting_time)
        text2 = (
            VGroup(
                Text(
                    "The anticipated distance", font_size=font_size_text, font=font
                ).set_color(BLUE),
                Text(
                    "- weights directional influence of neighbors and ",
                    font_size=font_size_text,
                    font=font,
                ),
                Text(
                    "- determines agent movement speed.",
                    font_size=font_size_text,
                    font=font,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .align_on_border(UP)
        )
        # these dots and lines to show again the diameter
        dot_right = Dot(circle.point_at_angle(0), color=BLUE)
        dot_left = Dot(circle.point_at_angle(PI), color=BLUE)

        line = Line(dot_left, dot_right)
        diameter_label = MathTex(r"l", font_size=28).next_to(line, DOWN * 2)

        self.play(Transform(title, text2))
        self.wait(waiting_time)
        self.play(
            FadeOut(
                s_line,
                interaction_line,
                proj_interaction_line1,
                proj_interaction_line2,
                interaction_line_anticipated,
                anticipated_circle1,
                anticipated_circle2,
                dashed_line1,
                dashed_line2,
                agent2_circle,
            ),
            FadeIn(dot_right, dot_left, line, diameter_label, run_time=2),
        )
        # self.wait(1)
        self.play(FadeOut(title, dot_right, dot_left, line, circle))

        # parameters direction function
        A = ValueTracker(1)
        B = ValueTracker(1)
        # parameters for distance function
        T = ValueTracker(1)
        v0 = ValueTracker(1)

        # speed plot
        axes1 = (
            setup_axes(x=4, y=2, xlabel="", ylabel="")
            .scale(0.4)
            .next_to(UP, RIGHT)
            .shift(RIGHT * 1.2)
        )
        # direction plot
        axes2 = (
            setup_axes(x=4, y=2, xlabel="", ylabel="")
            .scale(0.4)
            .next_to(UP, LEFT)
            .shift(LEFT * 1.2)
        )

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

        def exp_func(s):
            current_A = A.get_value()
            current_B = B.get_value()
            return current_A * np.exp(-s / current_B)

        distance_graph = always_redraw(
            lambda: axes1.plot(
                graph_func, x_range=[0, 4], use_smoothing=False, color=BLUE
            )
        )

        ylabel = MathTex(r"\overrightarrow{e_{ij}}", font_size=36).next_to(axes2, LEFT)
        vlabel = MathTex(f"v", font_size=36).next_to(axes1, LEFT)

        exp_graph = always_redraw(
            lambda: axes2.plot(
                exp_func, x_range=[0, 3], use_smoothing=False, color=BLUE
            )
        )
        x_range_center1 = (axes1.x_range[1] + axes1.x_range[0]) / 2
        x_range_center2 = (axes2.x_range[1] + axes2.x_range[0]) / 2
        text1 = Text(
            "Speed function",
            font_size=font_size_text,
            font=font,
        ).align_on_border(UP + RIGHT)
        text2 = Text(
            "Direction function",
            font_size=font_size_text,
            font=font,
        ).align_on_border(UP + LEFT)
        info_rectangle2 = Rectangle(
            width=3, height=1, color=WHITE, fill_opacity=0.2
        ).to_corner(m.DOWN * 2 + m.LEFT)
        info_text2 = always_redraw(
            lambda: Text(
                f"A: {A.get_value():.2f}\nB: {B.get_value():.2f}",
                font_size=20,
                font=font,
            ).move_to(info_rectangle2.get_center())
        )

        info_rectangle1 = Rectangle(
            width=3, height=1, color=WHITE, fill_opacity=0.2
        ).to_corner(m.DOWN * 2 + m.RIGHT)
        info_text1 = always_redraw(
            lambda: Text(
                f"T : {T.get_value():.2f}\nV0: {v0.get_value():.2f}",
                font_size=20,
                font=font,
            ).move_to(info_rectangle1.get_center())
        )

        self.play(
            Create(text1),
            Create(text2),
            interaction_label1.animate.next_to(axes1.c2p(x_range_center1, 0), DOWN),
            # circle and label move too
            interaction_label2.animate.next_to(axes2.c2p(x_range_center2, 0), DOWN),
            diameter_label.animate.next_to(axes1.c2p(1, 0), DOWN),
            FadeOut(interaction_label),
            Create(axes1),
            Create(axes2),
            Create(ylabel),
            Create(vlabel),
        )
        self.play(Create(exp_graph), Create(distance_graph))

        self.play(
            Create(info_rectangle1),
            Create(info_text1),
            Create(info_rectangle2),
            Create(info_text2),
        )

        speed_overlay = Rectangle(
            width=axes1.width + 1,
            height=axes1.height + 1,
            color=BLACK,
            fill_opacity=0.6,
        ).move_to(axes1)
        direction_overlay = Rectangle(
            width=axes2.width + 1,
            height=axes2.height + 1,
            color=BLACK,
            fill_opacity=0.6,
        ).move_to(axes2)

        # Add the overlay
        self.play(FadeIn(speed_overlay), run_time=1)
        for new_A in [1, 1.5, 1]:
            self.play(
                A.animate.set_value(new_A),  # Change the ValueTracker
                Circumscribe(info_text2[0], color=RED, time_width=0.1),
                run_time=2,
            )
        for new_B in [1, 0.5, 1]:
            self.play(
                B.animate.set_value(new_B),
                Circumscribe(info_text2[6], color=RED, time_width=0.1),
                run_time=2,
            )
        # Remove the overlay

        # Add the overlay
        self.play(FadeOut(speed_overlay), FadeIn(direction_overlay), run_time=1)

        for new_v0 in [1, 0.5, 1.5, 1]:
            self.play(
                v0.animate.set_value(new_v0),
                Circumscribe(info_text1[0], color=RED, time_width=0.1),
                run_time=2,
            )

        for new_T in [1, 0.5, 1.5, 1]:
            self.play(
                T.animate.set_value(new_T),
                Circumscribe(info_text1[6], color=RED, time_width=0.1),
                run_time=2,
            )

        self.play(FadeOut(direction_overlay), run_time=1)
        self.wait(1)
        self.play(FadeOut(*self.mobjects), run_time=5)

    def create_neighbors_act(self):
        text = Text(
            "Calculation of the movement direction.",
            font_size=font_size_text,
            font=font,
        ).align_on_border(UP)

        # Exit and agent setup
        exit_icon = ImageMobject("exit.png").scale(0.5)
        exit_position = exit_icon.get_center()
        anticipation_time = 0
        agent_radius = 0.5

        pos1 = np.array([-2, 0, 0])
        agent_circle = Circle(
            radius=agent_radius, color=YELLOW, fill_opacity=0.5
        ).move_to(pos1)

        # Positioning exit icon
        exit_icon.scale(0.3).next_to(agent_circle, RIGHT, buff=4.5)

        # Dashed line to exit
        dashed_line = DashedLine(
            start=agent_circle.get_center(),
            end=exit_icon.get_center(),
            color=WHITE,
            stroke_width=3,
        )

        # Direction to exit arrow
        direction_to_exit = exit_icon.get_center() - agent_circle.get_center()
        direction_to_exit /= np.linalg.norm(direction_to_exit)
        arrow_to_exit = Line(
            start=agent_circle.get_center(),
            end=agent_circle.get_center() + direction_to_exit,
            color=WHITE,
        ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)

        # Other agent positions
        agent_positions = (
            [agent_circle.get_center() + m.RIGHT * 2 + m.UP * 0.6]
            + [agent_circle.get_center() + m.LEFT * 3 + m.UP * 0.2]
            + [agent_circle.get_center() + m.UP * 1.0 + m.RIGHT * 0.7]
            # + [agent_circle.get_center() + m.UP * 1.6 + m.LEFT * 0.7]
            + [agent_circle.get_center() + m.LEFT * 1.5 + m.UP * 0.5]
            + [agent_circle.get_center() + RIGHT * 2 + m.DOWN * 1.2]
            + [agent_circle.get_center() + m.LEFT * 2 + m.DOWN * 0.7]
        )

        # Prepare agents with initial data
        agents = []
        for pos in [pos1] + agent_positions:
            direction_to_exit = (exit_position[:2] - pos[:2]) / np.linalg.norm(
                exit_position[:2] - pos[:2]
            )
            agents.append(
                {
                    "position": pos,
                    "radius": agent_radius,
                    "velocity": direction_to_exit * 0.5,
                    "anticipation_time": anticipation_time,
                    "orientation": direction_to_exit,
                    "destination": exit_position,
                    "strength": 2.0,
                    "range": 0.5,
                }
            )

        # Create other agent circles
        other_agent_circles = []
        other_agents = m.VGroup()
        for pos in agent_positions:
            new_agent = m.Circle(radius=agent_radius, color=m.BLUE, fill_opacity=0.5)
            new_agent.move_to(pos)
            other_agents.add(new_agent)
            other_agent_circles.append(new_agent)

        # ======================================= Animations ================================
        self.play(FadeIn(text))
        self.play(m.GrowFromCenter(agent_circle))
        text2 = Text(
            "Alone an agent would move straight to the exit.",
            font_size=font_size_text,
            font=font,
        ).align_on_border(UP)

        # Show exit and line to it
        self.play(m.FadeIn(exit_icon))
        self.play(Transform(text, text2))
        self.play(Create(dashed_line))
        self.play(m.FadeIn(arrow_to_exit), FadeOut(dashed_line))
        self.wait(1)
        # Add agents to the scene
        text2 = (
            VGroup(
                Text(
                    "Agent Path Deviation:", font_size=font_size_text, font=font
                ).set_color(BLUE),
                Text(
                    "In the presence of other agents,",
                    font_size=font_size_text,
                    font=font,
                ),
                Text(
                    "the original path must be adjusted.",
                    font_size=font_size_text,
                    font=font,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .align_on_border(UP)
        )

        self.play(Transform(text, text2))
        self.play(Transform(text, text2))
        self.wait(2)
        text2 = (
            VGroup(
                Text(
                    "Neighbor Influence:", font_size=font_size_text, font=font
                ).set_color(BLUE),
                Text("Influence is proportional", font_size=font_size_text, font=font),
                Text(
                    "to the distance between agents.",
                    font_size=font_size_text,
                    font=font,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .align_on_border(UP)
        )

        self.play(Transform(text, text2))
        self.wait(2)
        self.play(m.Create(other_agents), run_time=1)
        self.wait(2)
        agent1 = agents[0]
        total_influence = np.zeros(3)
        # perception_wedge
        text2 = Text(
            """
            Visual perception field.
            """,
            font_size=font_size_text,
            font=font,
        ).align_on_border(UP)
        perception_wedge = Sector(
            arc_center=pos1,
            radius=3,
            start_angle=-PI / 2,
            angle=PI,
            color=BLUE,
            fill_opacity=0.2,
        )
        self.play(Transform(text, text2), Create(perception_wedge))
        self.wait(1)

        # Compute influences and visualize
        info_rectangle = Rectangle(
            width=3, height=1, color=WHITE, fill_opacity=0.2
        ).to_corner(m.DOWN + m.RIGHT)
        distance = 0.0
        strength = 0.0
        info_text = always_redraw(
            lambda: Text(
                f"distance: {distance:.2f} m\nstrength: {strength:.2f}",
                font_size=20,
                font=font,
            ).move_to(info_rectangle.get_center())
        )
        self.play(Create(info_rectangle), Create(info_text))

        for i, agent2 in enumerate(agents[1:]):
            influence, adistance, distance, strength = neighbor_repulsion(
                agent1, agent2
            )
            total_influence += influence

            # Visualization of repulsion
            if np.linalg.norm(influence) > 0:
                text2 = Text(
                    """Neighbor has influence on the direction.
                    """,
                    font_size=font_size_text,
                    font=font,
                ).align_on_border(UP)

                self.play(Indicate(other_agent_circles[i]), color=m.ORANGE)
                # Highlight influencing agent
                self.play(
                    Transform(text, text2),
                )

                # Show influence arrow
                influence_arrow = Line(
                    start=agent1["position"],
                    end=agent1["position"] + influence,
                    color=ORANGE,
                ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
                # Draw dashed line between agents
                influence_dashed_line = m.DashedLine(
                    start=agent1["position"],
                    end=agent2["position"],
                    color=m.WHITE,
                    stroke_width=3,
                )
                self.play(m.Create(influence_dashed_line))
                self.play(m.Create(influence_arrow))
                # Remove highlighting and arrow
                self.play(m.FadeOut(influence_dashed_line, influence_arrow))
                # self.play(FadeOut(dashed_line))

            else:
                text2 = Text(
                    """
                Neighbor outside vision field.                
                """,
                    font_size=font_size_text,
                    font=font,
                ).align_on_border(UP)
                self.play(Indicate(other_agent_circles[i]), color=ORANGE)
                self.wait()
                self.play(
                    other_agent_circles[i]
                    .animate.set_fill(opacity=0.2)
                    .set_color(m.GREY),
                    Transform(text, text2),
                )
                self.wait(2)

        # Compute resulting direction
        desired_direction = (exit_position - agent1["position"]) / np.linalg.norm(
            exit_position - agent1["position"]
        )
        resulting_direction = (desired_direction + total_influence) / np.linalg.norm(
            desired_direction + total_influence
        )

        # Show resulting direction arrow
        orientation = resulting_direction[:2]  # Use 2D components
        orientation /= np.linalg.norm(orientation)  # Normalize

        resulting_arrow = m.Line(
            start=agent1["position"],
            end=agent1["position"] + resulting_direction,
            color=m.RED,
            buff=0,
        ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        # After computing resulting_direction
        agent1 = update_agent_state(agent1, resulting_direction, exit_position)
        orientation_arrow = m.Line(
            start=agent_circle.get_center(),
            end=agent_circle.get_center() + resulting_direction,
            color=m.RED,
            buff=0,
        ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        text2 = (
            VGroup(
                Text(
                    "Movement Direction:", font_size=font_size_text, font=font
                ).set_color(BLUE),
                Text(
                    "Sum of all influences determines",
                    font_size=font_size_text,
                    font=font,
                ),
                Text(
                    "the agent's movement trajectory.",
                    font_size=font_size_text,
                    font=font,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .align_on_border(UP)
        )
        self.play(
            FadeOut(arrow_to_exit),
            FadeOut(perception_wedge),
            m.Create(orientation_arrow),
            Transform(text, text2),
        )
        self.wait(3)
        # cleanup
        self.play(
            FadeOut(
                orientation_arrow,
                other_agents,
                info_rectangle,
                info_text,
                resulting_arrow,
                exit_icon,
                agent_circle,
                text,
            )
        )
        self.wait(1)

    # ----wall action
    def _setup_wall_visualization(self):
        wall_y = -1
        wall_start = np.array([-3, wall_y, 0])
        wall_end = np.array([3, wall_y, 0])
        wall = Line(wall_start, wall_end, color=WHITE)
        wall_buffer_distance = 0.5
        agent_radius = 0.2
        critical_distance = wall_buffer_distance + agent_radius
        influence_distance = 2 * critical_distance

        return (
            wall,
            wall_start,
            wall_end,
            wall_buffer_distance,
            agent_radius,
            critical_distance,
            influence_distance,
        )

    def _demonstrate_wall_interaction_cases(
        self,
        wall,
        wall_start,
        wall_end,
        wall_buffer_distance,
        agent_radius,
        critical_distance,
        influence_distance,
    ):
        def _define_bufferzon(
            influence_distance,
            critical_distance,
            wall_buffer_distance,
            agent_radius,
            wall_start,
            wall_end,
        ):
            # Dashed lines for buffer zones
            buffer_inner_line1 = DashedLine(
                start=wall_start + np.array([0, influence_distance, 0]),
                end=wall_end + np.array([0, influence_distance, 0]),
                color=BLUE,
                dash_length=0.1,
            )
            buffer_outer_line1 = DashedLine(
                start=wall_start - np.array([0, influence_distance, 0]),
                end=wall_end - np.array([0, influence_distance, 0]),
                color=BLUE,
                dash_length=0.1,
            )

            # Filled rectangle for influence zone
            buffer_fill1 = Polygon(
                wall_start + np.array([0, influence_distance, 0]),
                wall_end + np.array([0, influence_distance, 0]),
                wall_end - np.array([0, influence_distance, 0]),
                wall_start - np.array([0, influence_distance, 0]),
                color=BLUE,
                fill_opacity=0.1,
                stroke_width=0,
            )

            buffer_inner_line2 = DashedLine(
                start=wall_start + np.array([0, critical_distance, 0]),
                end=wall_end + np.array([0, critical_distance, 0]),
                color=ORANGE,
                dash_length=0.1,
            )
            buffer_outer_line2 = DashedLine(
                start=wall_start - np.array([0, critical_distance, 0]),
                end=wall_end - np.array([0, critical_distance, 0]),
                color=ORANGE,
                dash_length=0.1,
            )

            # Filled rectangle for critical zone
            buffer_fill2 = Polygon(
                wall_start + np.array([0, wall_buffer_distance + agent_radius, 0]),
                wall_end + np.array([0, critical_distance, 0]),
                wall_end - np.array([0, critical_distance, 0]),
                wall_start - np.array([0, critical_distance, 0]),
                color=ORANGE,
                fill_opacity=0.1,
                stroke_width=0,
            )
            return (
                buffer_fill1,
                buffer_fill2,
                buffer_inner_line1,
                buffer_outer_line1,
                buffer_inner_line2,
                buffer_outer_line2,
            )

        (
            buffer_fill1,
            buffer_fill2,
            buffer_inner_line1,
            buffer_outer_line1,
            buffer_inner_line2,
            buffer_outer_line2,
        ) = _define_bufferzon(
            influence_distance,
            critical_distance,
            wall_buffer_distance,
            agent_radius,
            wall_start,
            wall_end,
        )

        def update_arrow(agent_position, new_direction):
            return Arrow(
                start=np.array([agent_position[0], agent_position[1], 0]),
                end=np.array(
                    [
                        agent_position[0] + new_direction[0] * 0.8,
                        agent_position[1] + new_direction[1] * 0.8,
                        0,
                    ]
                ),
                color=YELLOW,
                buff=0,
                stroke_width=3,
            )

        def add_arrow_and_agent(agent_position, direction, radius, color=YELLOW):
            agent_circle = Circle(radius=radius, color=color, fill_opacity=0.8).move_to(
                agent_position
            )
            direction_arrow = Arrow(
                start=np.array([agent_position[0], agent_position[1], 0]),
                end=np.array(
                    [
                        agent_position[0] + direction[0] * 0.8,
                        agent_position[1] + direction[1] * 0.8,
                        0,
                    ]
                ),
                color=YELLOW,
                buff=0,
                stroke_width=3,
            )
            self.add(agent_circle, direction_arrow)
            return agent_circle, direction_arrow

        # Text setup
        starting_text = Text(
            "Wall influence", font=font, font_size=font_size_text
        ).align_on_border(UP)
        self.add(starting_text)

        # Visualization cases
        cases = [
            {
                "start_pos": np.array([-3, 1.5 * critical_distance, 0]),
                "direction": np.array([1, -1, 0]) / np.linalg.norm([1, -1, 0]),
                "text": """
                Agents within the influence zone are affected
                by the wall when moving toward it.
                """,
                "color": YELLOW,
                "special_effects": True,
            },
            {
                "start_pos": np.array([-3, wall_start[1] - 0.5 * critical_distance, 0]),
                "direction": np.array([1, 0, 0]) / np.linalg.norm([1, 0, 0]),
                "text": """
                In the critical zone, agents are
                pushed away.
                """,
                "color": YELLOW,
                "special_effects": True,
            },
            {
                "start_pos": np.array([-3, wall_start[1] + 1.3 * critical_distance, 0]),
                "direction": np.array([1, 0, 0]) / np.linalg.norm([1, 0, 0]),
                "text": """
                Agents outside the critical area and
                walking parallel to the wall
                experience no influence from the wall.
                """,
                "color": YELLOW,
                "special_effects": False,
            },
        ]
        self.add(wall)
        self.wait(2)
        for i, case in enumerate(cases):
            # Update text
            text_case = Text(
                case["text"], font=font, font_size=font_size_text
            ).align_on_border(UP)
            self.play(Transform(starting_text, text_case))
            if i == 0:
                # Add curly brace to represent the influence buffer
                curly_brace1 = BraceBetweenPoints(
                    point_1=wall_start + np.array([0, influence_distance, 0]),
                    point_2=wall_start - np.array([0, influence_distance, 0]),
                    direction=LEFT,
                )
                brace_label1 = Text(
                    "Influence Buffer", font_size=20, font=font
                ).next_to(curly_brace1, LEFT)
                self.play(
                    FadeIn(
                        buffer_inner_line1,
                        buffer_outer_line1,
                        buffer_fill1,
                        brace_label1,
                        curly_brace1,
                    )
                )
                self.wait(2)
            elif i == 1:
                # Add curly brace to represent the influence buffer
                curly_brace2 = BraceBetweenPoints(
                    point_1=wall_start + np.array([0, critical_distance, 0]),
                    point_2=wall_start - np.array([0, critical_distance, 0]),
                    direction=LEFT,
                )
                brace_label2 = Text("Critical Buffer", font_size=20, font=font).next_to(
                    curly_brace2, LEFT
                )
                self.play(
                    Transform(brace_label1, brace_label2),
                    Transform(curly_brace1, curly_brace2),
                )

                self.play(FadeIn(buffer_inner_line2, buffer_outer_line2, buffer_fill2))
                self.wait(2)
            # Agent setup
            agent_position = case["start_pos"].copy()
            direction = case["direction"]
            agent_circle, direction_arrow = add_arrow_and_agent(
                agent_position, direction, agent_radius, color=case["color"]
            )
            self.wait(2)
            influence_shown = False
            for _ in range(50):
                new_direction, what = calculate_new_direction(
                    direction,
                    agent_position,
                    agent_radius,
                    (wall_start, wall_end),
                    wall_buffer_distance,
                )
                agent_position[:2] += new_direction[:2] * 0.1
                agent_circle.move_to(agent_position)

                # Special effects
                if (
                    case["special_effects"]
                    and what == "influence"
                    and not influence_shown
                ):
                    influence_shown = True
                    self.play(
                        agent_circle.animate.scale(1.2).set_color(RED), run_time=0.5
                    )
                    self.play(
                        agent_circle.animate.scale(1 / 1.2).set_color(YELLOW),
                        run_time=0.5,
                    )
                    self.wait(0.2)

                direction_arrow.become(update_arrow(agent_position, new_direction))
                self.wait(0.1)

            # Cleanup
            self.play(FadeOut(agent_circle, direction_arrow))

        # Final cleanup
        self.play(
            FadeOut(
                starting_text,
                curly_brace1,
                brace_label1,
                buffer_inner_line2,
                buffer_inner_line1,
                buffer_outer_line1,
                buffer_outer_line2,
                wall,
            )
        )
        self.wait(2)

    def create_wall_act(self):
        wall_params = self._setup_wall_visualization()
        self._demonstrate_wall_interaction_cases(*wall_params)

    def simulation_act1(self):
        grid = NumberPlane(
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.5,
            },
        )
        self.play(Create(grid), run_time=2)

        agent_radius = 0.5

        pos1 = np.array([-2, 0, 0])
        agent1 = {
            "position": pos1,
            "radius": agent_radius,
            "velocity": np.array([1, 0, 0]),
            "orientation": np.array([1, 0, 0]),
            "destination": np.array([4, 0, 0]),
            "strength": 1.0,
            "anticipation_time": 1,
            "range": 0.5,
        }

        pos2 = np.array([2, 0, 0])
        agent2 = {
            "position": pos2,
            "radius": agent_radius,
            "velocity": np.array([0, 0, 0]),
            "orientation": np.array([1, 0, 0]),
            "destination": np.array([4, 0, 0]),
            "strength": 1.0,
            "anticipation_time": 1,
            "range": 2.5,
        }
        agent_circle1 = Circle(
            radius=agent_radius, color=BLUE, fill_opacity=0.5
        ).move_to(pos1)
        agent_circle2 = Circle(
            radius=agent_radius, color=GREY, fill_opacity=0.5
        ).move_to(pos2)
        dashed_line = DashedLine(start=pos1, end=pos2, color=WHITE, stroke_width=0.5)
        desired_direction = agent1["orientation"]
        # ------------ visualisation --------------
        text = Text(
            """Static agent""",
            font_size=font_size_text,
            font=font,
        ).align_on_border(UP + LEFT)
        influence = 0
        resulting_direction = (desired_direction + influence) / np.linalg.norm(
            desired_direction + influence
        )
        direction_arrow = always_redraw(
            lambda: m.Line(
                start=agent1["position"],
                end=agent1["position"] + resulting_direction,
                color=m.BLUE,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        )

        self.play(
            Write(text),
            m.GrowFromCenter(agent_circle1),
            m.GrowFromCenter(agent_circle2),
            FadeIn(dashed_line),
        )
        self.play(Create(direction_arrow))
        current_pos = pos1
        info_rectangle = Rectangle(
            width=3, height=1, color=WHITE, fill_opacity=0.2
        ).to_corner(m.DOWN * 2.8 + m.LEFT)
        # Dynamically updating info_text
        distance = 0.0
        strength = 0.0
        info_text = always_redraw(
            lambda: Text(
                f"distance: {distance:.2f} m\nstrength: {strength:.2f}",
                font_size=20,
                font=font,
            ).move_to(info_rectangle.get_center())
        )
        self.play(Create(info_rectangle), Create(info_text))

        for frame in range(40):
            influence, adistance, distance, strength = neighbor_repulsion(
                agent1, agent2
            )

            # Update velocity based on influence
            resulting_direction = (desired_direction + influence) / np.linalg.norm(
                desired_direction + influence
            )

            # Move agent incrementally

            current_pos = current_pos.astype(float) + 0.2 * np.array(
                resulting_direction, dtype=float
            )
            agent1["position"] = current_pos

            color = GREY if strength < 0.1 else YELLOW

            # Update visualization
            self.play(
                agent_circle1.animate.move_to(current_pos),
                dashed_line.animate.put_start_and_end_on(current_pos, pos2),
                agent_circle2.animate.set_fill(opacity=0.2).set_color(color),
                run_time=0.1,
            )

        self.wait(1)
        self.play(
            FadeOut(
                agent_circle1,
                agent_circle2,
                info_rectangle,
                info_text,
                dashed_line,
                direction_arrow,
                text,
            )
        )
        self.wait(1)

    def simulation_act2(self):
        # grid = NumberPlane(
        #     background_line_style={
        #         "stroke_color": GREY,
        #         "stroke_width": 1,
        #         "stroke_opacity": 0.5,
        #     },
        # )
        # self.play(Create(grid), run_time=2)

        # Wait to show the grid
        self.wait(2)
        agent_radius = 0.5

        pos1 = np.array([-4, 0, 0])
        agent1 = {
            "position": pos1,
            "radius": agent_radius,
            "velocity": np.array([1, 0, 0]),
            "orientation": np.array([1, 0, 0]),
            "destination": np.array([4, 0, 0]),
            "strength": 1,
            "anticipation_time": 0.7,
            "range": 0.5,
        }

        pos2 = np.array([4, 0, 0])
        agent2 = {
            "position": pos2,
            "radius": agent_radius,
            "velocity": np.array([-1, 0, 0]),
            "orientation": np.array([-1, 0, 0]),
            "destination": np.array([-4, 0, 0]),
            "strength": 1,
            "anticipation_time": 0.9,
            "range": 0.5,
        }
        agent_circle1 = Circle(
            radius=agent_radius, color=BLUE, fill_opacity=0.5
        ).move_to(pos1)
        agent_circle2 = Circle(
            radius=agent_radius, color=YELLOW, fill_opacity=0.5
        ).move_to(pos2)
        dashed_line = DashedLine(start=pos1, end=pos2, color=WHITE, stroke_width=0.5)
        desired_direction1 = agent1["orientation"]
        desired_direction2 = agent2["orientation"]
        # ------------ visualisation --------------
        text = Text(
            """Head-on""",
            font_size=font_size_text,
            font=font,
        ).align_on_border(UP + LEFT)

        self.play(
            Write(text),
            m.GrowFromCenter(agent_circle1),
            m.GrowFromCenter(agent_circle2),
            FadeIn(dashed_line),
        )
        current_pos1 = pos1
        current_pos2 = pos2
        info_rectangle = Rectangle(
            width=3, height=1, color=WHITE, fill_opacity=0.2
        ).to_corner(m.DOWN * 2.8 + m.LEFT)
        # Dynamically updating info_text
        distance = 0.0
        strength = 0.0
        info_text = always_redraw(
            lambda: Text(
                f"distance: {distance:.2f} m\nstrength: {strength:.2f}",
                font_size=20,
                font=font,
            ).move_to(info_rectangle.get_center())
        )
        self.play(Create(info_text), Create(info_rectangle))
        influence = 0
        influence2 = 0
        resulting_direction1 = (desired_direction1 + influence) / np.linalg.norm(
            desired_direction1 + influence
        )
        direction_arrow1 = always_redraw(
            lambda: m.Line(
                start=agent1["position"],
                end=agent1["position"] + resulting_direction1,
                color=m.BLUE,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        )
        resulting_direction2 = (desired_direction2 + influence2) / np.linalg.norm(
            desired_direction1 + influence
        )
        direction_arrow2 = always_redraw(
            lambda: m.Line(
                start=agent2["position"],
                end=agent2["position"] + resulting_direction2,
                color=m.YELLOW,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        )
        self.play(Create(direction_arrow1), Create(direction_arrow2))
        for frame in range(35):
            influence, adistance, distance, strength = neighbor_repulsion(
                agent1, agent2
            )
            influence2, adistance2, distance2, strength2 = neighbor_repulsion(
                agent2, agent1
            )

            # Update velocity based on influence
            resulting_direction1 = (desired_direction1 + influence) / np.linalg.norm(
                desired_direction1 + influence
            )
            resulting_direction2 = (desired_direction2 + influence2) / np.linalg.norm(
                desired_direction2 + influence2
            )
            resulting_arrow1 = m.Line(
                start=agent1["position"],
                end=agent1["position"] + resulting_direction1,
                color=m.RED,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)

            # Move agent incrementally

            current_pos1 = current_pos1.astype(float) + 0.2 * np.array(
                resulting_direction1, dtype=float
            )
            agent1["position"] = current_pos1

            current_pos2 = current_pos2.astype(float) + 0.2 * np.array(
                resulting_direction2, dtype=float
            )
            agent2["position"] = current_pos2

            color = GREY if strength < 0.1 else YELLOW

            # Update visualization
            self.play(
                agent_circle1.animate.move_to(current_pos1),
                agent_circle2.animate.move_to(current_pos2),
                dashed_line.animate.put_start_and_end_on(current_pos1, current_pos2),
                run_time=0.1,
            )

        self.wait(1)
        self.play(
            FadeOut(
                agent_circle1,
                agent_circle2,
                info_rectangle,
                info_text,
                dashed_line,
                direction_arrow1,
                direction_arrow2,
                text,
                # grid,
            )
        )
        self.wait(1)

    def simulation_act3(self):
        # grid = NumberPlane(
        #     background_line_style={
        #         "stroke_color": GREY,
        #         "stroke_width": 1,
        #         "stroke_opacity": 0.5,
        #     },
        # )
        # self.play(Create(grid), run_time=2)
        agent_radius = 0.5

        pos1 = np.array([-4, 0, 0])
        agent1 = {
            "position": pos1,
            "radius": agent_radius,
            "velocity": np.array([1, 0, 0]),
            "orientation": np.array([1, 0, 0]),
            "destination": np.array([4, 0, 0]),
            "strength": 1,
            "anticipation_time": 0.7,
            "range": 0.5,
        }

        pos2 = np.array([0, -4, 0])
        agent2 = {
            "position": pos2,
            "radius": agent_radius,
            "velocity": np.array([0, 1, 0]),
            "orientation": np.array([0, 1, 0]),
            "destination": np.array([0, 4, 0]),
            "strength": 1,
            "anticipation_time": 0.9,
            "range": 0.5,
        }
        agent_circle1 = Circle(
            radius=agent_radius, color=BLUE, fill_opacity=0.5
        ).move_to(pos1)
        agent_circle2 = Circle(
            radius=agent_radius, color=YELLOW, fill_opacity=0.5
        ).move_to(pos2)
        dashed_line = DashedLine(start=pos1, end=pos2, color=WHITE, stroke_width=0.5)
        desired_direction1 = agent1["orientation"]
        desired_direction2 = agent2["orientation"]
        # ------------ visualisation --------------
        text = Text(
            """Crossing
            """,
            font_size=font_size_text,
            font=font,
        ).align_on_border(UP + LEFT)

        self.play(
            Write(text),
            m.GrowFromCenter(agent_circle1),
            m.GrowFromCenter(agent_circle2),
            FadeIn(dashed_line),
        )
        current_pos1 = pos1
        current_pos2 = pos2
        info_rectangle = Rectangle(
            width=3, height=1, color=WHITE, fill_opacity=0.2
        ).to_corner(m.DOWN * 2.8 + m.LEFT)
        # Dynamically updating info_text
        distance = 0.0
        strength = 0.0
        info_text = always_redraw(
            lambda: Text(
                f"distance: {distance:.2f} m\nstrength: {strength:.2f}",
                font_size=20,
                font=font,
            ).move_to(info_rectangle.get_center())
        )
        self.play(Create(info_text), Create(info_rectangle))
        influence = 0
        influence2 = 0
        resulting_direction1 = (desired_direction1 + influence) / np.linalg.norm(
            desired_direction1 + influence
        )
        direction_arrow1 = always_redraw(
            lambda: m.Line(
                start=agent1["position"],
                end=agent1["position"] + resulting_direction1,
                color=m.BLUE,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        )
        resulting_direction2 = (desired_direction2 + influence2) / np.linalg.norm(
            desired_direction1 + influence
        )
        direction_arrow2 = always_redraw(
            lambda: m.Line(
                start=agent2["position"],
                end=agent2["position"] + resulting_direction2,
                color=m.YELLOW,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        )
        self.play(Create(direction_arrow1), Create(direction_arrow2))
        for frame in range(35):
            influence, adistance, distance, strength = neighbor_repulsion(
                agent1, agent2
            )
            influence2, adistance2, distance2, strength2 = neighbor_repulsion(
                agent2, agent1
            )

            # Update velocity based on influence
            resulting_direction1 = (desired_direction1 + influence) / np.linalg.norm(
                desired_direction1 + influence
            )
            resulting_direction2 = (desired_direction2 + influence2) / np.linalg.norm(
                desired_direction2 + influence2
            )
            resulting_arrow1 = m.Line(
                start=agent1["position"],
                end=agent1["position"] + resulting_direction1,
                color=m.RED,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)

            # Move agent incrementally

            current_pos1 = current_pos1.astype(float) + 0.2 * np.array(
                resulting_direction1, dtype=float
            )
            agent1["position"] = current_pos1

            current_pos2 = current_pos2.astype(float) + 0.2 * np.array(
                resulting_direction2, dtype=float
            )
            agent2["position"] = current_pos2

            color = GREY if strength < 0.1 else YELLOW

            # Update visualization
            self.play(
                agent_circle1.animate.move_to(current_pos1),
                agent_circle2.animate.move_to(current_pos2),
                dashed_line.animate.put_start_and_end_on(current_pos1, current_pos2),
                run_time=0.1,
            )

        self.wait(1)
        self.play(
            FadeOut(
                agent_circle1,
                agent_circle2,
                info_rectangle,
                info_text,
                dashed_line,
                direction_arrow1,
                direction_arrow2,
                text,
                # grid,
            )
        )
        self.wait(1)

    def simulation_act4(self):
        grid = NumberPlane(
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.5,
            },
        )
        self.play(Create(grid), run_time=1)
        agent_radius = 0.5

        pos1 = np.array([-3, 0, 0])
        agent1 = {
            "position": pos1,
            "radius": agent_radius,
            "velocity": np.array([1, 0, 0]),
            "orientation": np.array([1, 0, 0]),
            "destination": np.array([4, 0, 0]),
            "strength": 1,
            "anticipation_time": 0.7,
            "range": 0.5,
        }

        pos2 = np.array([0, -3, 0])
        agent2 = {
            "position": pos2,
            "radius": agent_radius,
            "velocity": np.array([0, 1, 0]),
            "orientation": np.array([0, 1, 0]),
            "destination": np.array([0, 4, 0]),
            "strength": 1,
            "anticipation_time": 0.9,
            "range": 0.5,
        }
        agent_circle1 = Circle(
            radius=agent_radius, color=BLUE, fill_opacity=0.5
        ).move_to(pos1)
        agent_circle2 = Circle(
            radius=agent_radius, color=YELLOW, fill_opacity=0.5
        ).move_to(pos2)
        pos3 = np.array([0, 3, 0])
        agent3 = {
            "position": pos3,
            "radius": agent_radius,
            "velocity": np.array([0, -1, 0]),
            "orientation": np.array([0, -1, 0]),
            "destination": np.array([0, -4, 0]),
            "strength": 1,
            "anticipation_time": 0.7,
            "range": 0.5,
        }

        pos4 = np.array([3, 0, 0])
        agent4 = {
            "position": pos4,
            "radius": agent_radius,
            "velocity": np.array([-1, 0, 0]),
            "orientation": np.array([-1, 0, 0]),
            "destination": np.array([-4, 0, 0]),
            "strength": 1,
            "anticipation_time": 0.9,
            "range": 0.5,
        }
        agent_circle1 = Circle(
            radius=agent_radius, color=YELLOW, fill_opacity=0.5
        ).move_to(pos1)
        agent_circle2 = Circle(
            radius=agent_radius, color=BLUE, fill_opacity=0.5
        ).move_to(pos2)
        agent_circle3 = Circle(
            radius=agent_radius, color=RED, fill_opacity=0.5
        ).move_to(pos3)
        agent_circle4 = Circle(
            radius=agent_radius, color=GREEN, fill_opacity=0.5
        ).move_to(pos4)

        desired_direction1 = agent1["orientation"]
        desired_direction2 = agent2["orientation"]
        desired_direction3 = agent3["orientation"]
        desired_direction4 = agent4["orientation"]

        # ------------ visualisation --------------
        text = Text(
            """Crossing 4
            """,
            font_size=font_size_text,
            font=font,
        ).align_on_border(UP + LEFT)

        self.play(
            Write(text),
            m.GrowFromCenter(agent_circle1),
            m.GrowFromCenter(agent_circle2),
            m.GrowFromCenter(agent_circle3),
            m.GrowFromCenter(agent_circle4),
        )
        current_pos1 = pos1
        current_pos2 = pos2
        current_pos3 = pos3
        current_pos4 = pos4

        info_rectangle = Rectangle(
            width=3, height=1, color=WHITE, fill_opacity=0.2
        ).to_corner(m.DOWN * 2.8 + m.LEFT)
        # Dynamically updating info_text
        influence = 0
        influence2 = 0
        influenc3 = 0
        influence4 = 0
        resulting_direction1 = (desired_direction1 + influence) / np.linalg.norm(
            desired_direction1 + influence
        )
        direction_arrow1 = always_redraw(
            lambda: m.Line(
                start=agent1["position"],
                end=agent1["position"] + resulting_direction1,
                color=m.YELLOW,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        )
        resulting_direction2 = (desired_direction2 + influence2) / np.linalg.norm(
            desired_direction2 + influence2
        )
        direction_arrow2 = always_redraw(
            lambda: m.Line(
                start=agent2["position"],
                end=agent2["position"] + resulting_direction2,
                color=m.BLUE,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        )

        resulting_direction3 = (desired_direction3 + influenc3) / np.linalg.norm(
            desired_direction3 + influenc3
        )
        direction_arrow3 = always_redraw(
            lambda: m.Line(
                start=agent3["position"],
                end=agent3["position"] + resulting_direction3,
                color=m.RED,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        )
        resulting_direction4 = (desired_direction4 + influence4) / np.linalg.norm(
            desired_direction4 + influence4
        )
        direction_arrow4 = always_redraw(
            lambda: m.Line(
                start=agent4["position"],
                end=agent4["position"] + resulting_direction4,
                color=m.GREEN,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
        )

        self.play(
            Create(direction_arrow1),
            Create(direction_arrow2),
            Create(direction_arrow3),
            Create(direction_arrow4),
        )

        for frame in range(35):
            # agent 1
            # Compute total influence for each agent
            influence = compute_total_influence(agent1, [agent2, agent3, agent4])
            influence2 = compute_total_influence(agent2, [agent1, agent3, agent4])
            influence3 = compute_total_influence(agent3, [agent1, agent2, agent4])
            influence4 = compute_total_influence(agent4, [agent1, agent2, agent3])

            # Update velocity based on influence
            resulting_direction1 = (desired_direction1 + influence) / np.linalg.norm(
                desired_direction1 + influence
            )
            # agent 2

            # Update velocity based on influence
            resulting_direction2 = (desired_direction2 + influence2) / np.linalg.norm(
                desired_direction2 + influence2
            )
            # agent 3

            # Update velocity based on influence
            resulting_direction3 = (desired_direction3 + influence3) / np.linalg.norm(
                desired_direction3 + influence3
            )
            # agent 4

            # Update velocity based on influence
            resulting_direction4 = (desired_direction4 + influence4) / np.linalg.norm(
                desired_direction4 + influence4
            )

            resulting_arrow1 = m.Line(
                start=agent1["position"],
                end=agent1["position"] + resulting_direction1,
                color=m.RED,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
            resulting_arrow2 = m.Line(
                start=agent2["position"],
                end=agent2["position"] + resulting_direction2,
                color=m.RED,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
            resulting_arrow3 = m.Line(
                start=agent3["position"],
                end=agent3["position"] + resulting_direction3,
                color=m.RED,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)
            resulting_arrow4 = m.Line(
                start=agent4["position"],
                end=agent4["position"] + resulting_direction4,
                color=m.RED,
                buff=0,
            ).add_tip(tip_shape=StealthTip, tip_length=0.1, tip_width=0.5)

            # Move agent incrementally
            dt = 0.2
            current_pos1 = current_pos1.astype(float) + dt * np.array(
                resulting_direction1, dtype=float
            )
            agent1["position"] = current_pos1

            current_pos2 = current_pos2.astype(float) + dt * np.array(
                resulting_direction2, dtype=float
            )
            agent2["position"] = current_pos2

            current_pos3 = current_pos3.astype(float) + dt * np.array(
                resulting_direction3, dtype=float
            )
            agent3["position"] = current_pos3

            current_pos4 = current_pos4.astype(float) + dt * np.array(
                resulting_direction4, dtype=float
            )
            agent4["position"] = current_pos4

            # Update visualization
            self.play(
                agent_circle1.animate.move_to(current_pos1),
                agent_circle2.animate.move_to(current_pos2),
                agent_circle3.animate.move_to(current_pos3),
                agent_circle4.animate.move_to(current_pos4),
                run_time=0.1,
            )

        self.wait(1)
        self.play(
            FadeOut(
                agent_circle1,
                agent_circle2,
                agent_circle3,
                agent_circle4,
                direction_arrow1,
                direction_arrow2,
                direction_arrow3,
                direction_arrow4,
                text,
                grid,
            )
        )
        self.wait(1)

    # ===============================================================
    def construct(self):
        self.ShowIntro()
        self.create_predicted_distance_act()
        self.create_neighbors_act()
        self.create_wall_act()
        self.simulation_act1()
        self.simulation_act2()
        self.simulation_act3()
        self.simulation_act4()
