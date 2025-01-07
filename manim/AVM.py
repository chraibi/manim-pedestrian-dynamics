"""
Manim animation for visualising AVM.
Run with: manim -pqh change_v0_T.py
"""

from manim import *
import numpy as np
from manim.utils.rate_functions import smooth

from dataclasses import dataclass

# Global variable for text font
font = "Inconsolata-dz for Powerline"
font_size_text = 30


@dataclass
class VisualizationConfig:
    CalculateInfluenceDirection = True


class AVM(Scene):
    def __init__(
        self, *args, config: VisualizationConfig = VisualizationConfig(), **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config

    def ShowCalculateInfluenceDirection(self):
        # Step 1: Define desired and predicted directions
        desired_direction = np.array([2, 1, 0])
        predicted_direction = np.array([1, 2, 0])

        # Normalize directions for consistency
        desired_direction = desired_direction / np.linalg.norm(desired_direction)
        predicted_direction = predicted_direction / np.linalg.norm(predicted_direction)

        # Step 2: Compute orthogonal direction
        orthogonal_direction = np.array(
            [-desired_direction[1], desired_direction[0], 0]
        )
        orthogonal_direction = orthogonal_direction / np.linalg.norm(
            orthogonal_direction
        )

        # Step 3: Compute alignment (scalar product)
        alignment = np.dot(orthogonal_direction, predicted_direction)

        # Step 4: Compute influence direction
        influence_direction = orthogonal_direction
        if np.abs(alignment) < 1e-8:  # Near zero
            influence_direction = (
                -orthogonal_direction
                if np.random.randint(0, 2) == 0
                else orthogonal_direction
            )
        elif alignment > 0:
            influence_direction = -orthogonal_direction

        # Visualization
        # Axes
        axes = Axes(
            x_range=[-2, 2],
            y_range=[-2, 2],
            axis_config={"include_tip": True, "stroke_width": 2},
        ).add_coordinates()
        self.play(Create(axes))

        # Desired direction
        desired_vector = Vector(desired_direction, color=BLUE).shift(axes.c2p(0, 0))
        desired_label = MathTex(r"\text{Desired}", font_size=24).next_to(
            desired_vector, RIGHT, buff=0.1
        )
        self.play(Create(desired_vector), Write(desired_label))

        # Predicted direction
        predicted_vector = Vector(predicted_direction, color=GREEN).shift(
            axes.c2p(0, 0)
        )
        predicted_label = MathTex(r"\text{Predicted}", font_size=24).next_to(
            predicted_vector, UP + RIGHT, buff=0.2
        )
        self.play(Create(predicted_vector), Write(predicted_label))

        # Orthogonal direction
        orthogonal_vector = Vector(orthogonal_direction, color=YELLOW).shift(
            axes.c2p(0, 0)
        )
        orthogonal_label = MathTex(r"\text{Orthogonal}", font_size=24).next_to(
            orthogonal_vector, LEFT
        )
        self.play(Create(orthogonal_vector), Write(orthogonal_label))

        # Show alignment value
        alignment_label = MathTex(
            f"\\text{{Alignment: }} {alignment:.2f}", font_size=28
        ).to_corner(UP + LEFT)
        self.play(Write(alignment_label))

        # Influence direction
        influence_vector = Vector(influence_direction, color=RED).shift(axes.c2p(0, 0))
        influence_label = MathTex(r"\text{Influence}", font_size=24).next_to(
            influence_vector, RIGHT, buff=0.5
        )
        self.play(Create(influence_vector), Write(influence_label))

        self.wait(3)

    def construct(self):
        # Modular setup with stage control
        if self.config.CalculateInfluenceDirection:
            self.ShowCalculateInfluenceDirection()
