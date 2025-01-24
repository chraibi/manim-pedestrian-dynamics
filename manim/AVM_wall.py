from manim import *
import numpy as np

font = "Inconsolata-dz for Powerline"
font_size_text = 30


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


class WallInteractionScene(Scene):
    def construct(self):
        self._visualize_wall_interaction()

    def _visualize_wall_interaction(self):
        wall_params = self._setup_wall_visualization()
        self._demonstrate_wall_interaction_cases(*wall_params)

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
                wall_start + np.array([0, wall_buffer_distance + agent_radius, 0]),
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
                color=GREEN,
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
                color=GREEN,
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
            },
            {
                "start_pos": np.array([-3, wall_start[1] - 0.5 * critical_distance, 0]),
                "direction": np.array([1, 0, 0]) / np.linalg.norm([1, 0, 0]),
                "text": """
                In the critical zone, agents are
                pushed away.
                """,
                "color": YELLOW,
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
                brace_label1 = Text("Influence Buffer", font_size=20).next_to(
                    curly_brace1, LEFT
                )
                self.add(
                    buffer_inner_line1,
                    buffer_outer_line1,
                    buffer_fill1,
                    brace_label1,
                    curly_brace1,
                )
            else:
                # Add curly brace to represent the influence buffer
                curly_brace2 = BraceBetweenPoints(
                    point_1=wall_start + np.array([0, critical_distance, 0]),
                    point_2=wall_start - np.array([0, critical_distance, 0]),
                    direction=LEFT,
                )
                brace_label2 = Text("Critical Buffer", font_size=20).next_to(
                    curly_brace2, LEFT
                )
                self.play(
                    Transform(brace_label1, brace_label2),
                    Transform(curly_brace1, curly_brace2),
                )

                self.add(
                    buffer_inner_line2,
                    buffer_outer_line2,
                    buffer_fill2,
                )
            self.wait(1)

            # Agent setup
            agent_position = case["start_pos"].copy()
            direction = case["direction"]
            agent_circle, direction_arrow = add_arrow_and_agent(
                agent_position, direction, agent_radius, color=case["color"]
            )

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
                if what == "influence" and not influence_shown:
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
        self.play(FadeOut(starting_text))
