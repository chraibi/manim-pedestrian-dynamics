from manim import *
import numpy as np
import manim as m


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
    if not in_perception_range:
        return np.zeros(3)  # Return a 3D zero vector for consistency

    # Compute S_Gap and R_dist
    s_gap = np.dot(velocity1[:2] - velocity2[:2], ep12) * anticipation_time
    r_dist = max(adjusted_dist - s_gap, 0)

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
    return to_3d(influence_direction * interaction_strength)


def calculate_influence_direction(d1, newep12):
    """
    Calculate the adjusted influence direction based on desired and predicted directions.
    """
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
    def create_predicted_distance_scene(
        self,
        pos1=np.array([-2, 0, 0]),
        velocity1=np.array([1, 1, 0]),
        pos2=np.array([2, 0, 0]),
        velocity2=np.array([-1, 2, 0]),
        anticipation_time=1.0,
        radius=0.4,
        velocity_scale=0.5,
    ):
        title = Text("Predicted Distance with Time constant", font_size=36).to_edge(UP)
        self.play(Write(title))
        # Parameters for agents
        anticipation_time = 1.0
        radius = 0.4
        velocity_scale = 0.5
        influence_scale = 0.5

        # Agent 1 (i)
        pos1 = np.array([-2, 0, 0])
        velocity1 = np.array([1, 1, 0])
        destination1 = np.array([3, 2, 0])

        # Agent 2 (j)
        pos2 = np.array([2, 0, 0])
        velocity2 = np.array([-1, 2, 0])

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
            stroke_width=1,
        )
        dashed_line2 = DashedLine(
            start=pos2,
            end=anticipated_pos2,
            color=RED,
            dash_length=0.1,
            stroke_width=1,
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
        interaction_label = MathTex("s_{i,j}^a(t + t^a)", font_size=30).move_to(
            s_line.get_center() + DOWN * 0.5
        )
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

        self.play(GrowFromCenter(agent1_circle), GrowFromCenter(agent2_circle))
        self.play(Create(interaction_line))
        # Animations to move circles from original to anticipated positions
        self.play(
            Create(dashed_line1),
            Create(dashed_line2),
            anticipated_circle1.animate.move_to(anticipated_pos1),
            anticipated_circle2.animate.move_to(anticipated_pos2),
        )

        self.play(Create(interaction_line_anticipated))
        self.play(Create(proj_interaction_line1), Create(proj_interaction_line2))
        self.play(FadeIn(s_line), FadeIn(interaction_label))

    def create_visibility_scene(self):
        # Exit and agent setup
        exit_icon = ImageMobject("exit.png").scale(0.5)
        exit_position = exit_icon.get_center()
        anticipation_time = 1
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
            + [agent_circle.get_center() + m.UP * 1.0 + m.RIGHT * 0.7]
            + [agent_circle.get_center() + m.UP * 1.6 + m.LEFT * 0.7]
            + [agent_circle.get_center() + m.LEFT * 1.5 + m.UP * 0.5]
            + [agent_circle.get_center() + m.LEFT * 2 + m.DOWN * 0.7]
            + [agent_circle.get_center() + m.LEFT * 3 + m.UP * 0.2]
            + [agent_circle.get_center() + RIGHT * 2 + m.DOWN * 1.2]
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
                    "strength": 1.0,
                    "range": 2.0,
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

        # Animations
        self.play(m.GrowFromCenter(agent_circle))
        self.wait(2)
        self.play(m.Create(other_agents), run_time=1)
        self.wait(2)
        self.play(m.FadeIn(exit_icon))
        self.add(dashed_line)
        self.play(m.FadeIn(arrow_to_exit))
        agent1 = agents[0]
        # Simulation loop
        for ii in range(1):
            agent1["position"] = pos1
            total_influence = np.zeros(3)

            # Compute influences and visualize
            for i, agent2 in enumerate(agents[1:]):
                influence = neighbor_repulsion(agent1, agent2)
                total_influence += influence

                # Visualization of repulsion
                if np.linalg.norm(influence) > 0:
                    # Highlight influencing agent
                    self.play(
                        other_agent_circles[i]
                        .animate.set_fill(opacity=0.8)
                        .set_color(m.ORANGE)
                    )

                    # Show influence arrow
                    influence_arrow = m.Arrow(
                        start=agent1["position"],
                        end=agent1["position"] + influence,
                        color=m.ORANGE,
                        buff=0,
                    )
                    # Draw dashed line between agents
                    influence_dashed_line = m.DashedLine(
                        start=agent1["position"],
                        end=agent2["position"],
                        color=m.WHITE,
                        stroke_width=3,
                    )
                    if ii == 0:
                        self.play(m.Create(influence_dashed_line))
                        self.play(m.Create(influence_arrow))
                        self.wait(0.01)
                        # Remove highlighting and arrow
                        self.play(m.FadeOut(influence_dashed_line, influence_arrow))
                        # self.play(FadeOut(dashed_line))

                    self.play(
                        other_agent_circles[i]
                        .animate.set_fill(opacity=0.5)
                        .set_color(m.BLUE),
                    )
                else:
                    self.play(
                        other_agent_circles[i]
                        .animate.set_fill(opacity=0.2)
                        .set_color(m.GREY),
                    )

            # Compute resulting direction
            desired_direction = (exit_position - agent1["position"]) / np.linalg.norm(
                exit_position - agent1["position"]
            )
            resulting_direction = (
                desired_direction + total_influence
            ) / np.linalg.norm(desired_direction + total_influence)

            # Show resulting direction arrow
            orientation = resulting_direction[:2]  # Use 2D components
            orientation /= np.linalg.norm(orientation)  # Normalize

            # Create or update orientation arrow
            new_orientation_arrow = m.Arrow(
                start=agent_circle.get_center(),
                end=agent_circle.get_center()
                + np.append(orientation, 0),  # Add z-component
                color=m.GREEN,
                buff=0,
            )
            resulting_arrow = m.Arrow(
                start=agent1["position"],
                end=agent1["position"] + resulting_direction,
                color=m.RED,
                buff=0,
            )
            # After computing resulting_direction
            agent1 = update_agent_state(agent1, resulting_direction, exit_position)
            orientation_arrow = m.Arrow(
                start=agent_circle.get_center(),
                end=agent_circle.get_center() + resulting_direction,
                color=m.YELLOW,
                buff=0,
            )
            if ii == 0:
                self.play(m.Create(orientation_arrow))

            new_arrow_to_exit = m.Line(
                start=agent_circle.get_center(),
                end=agent_circle.get_center() + desired_direction,
                color=m.WHITE,
            ).add_tip(tip_shape=m.StealthTip, tip_length=0.1, tip_width=0.5)

            # Animate arrow update
            self.play(m.Transform(arrow_to_exit, new_arrow_to_exit))
            # Update position
            pos1 = np.array(pos1, dtype=float)
            pos1[:2] += resulting_direction[:2] * 0.5
            # agent_circle.move_to(pos1)
            self.play(m.FadeOut(resulting_arrow))

    def construct(self):
        # self.create_predicted_distance_scene()
        self.create_visibility_scene()
