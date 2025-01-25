from manim import *
import numpy as np


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
    Calculate the adjusted influence direction.
    """
    if np.linalg.norm(newep12) > 0:
        return newep12 / np.linalg.norm(newep12)
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

    def construct(self):
        self.create_predicted_distance_scene()
