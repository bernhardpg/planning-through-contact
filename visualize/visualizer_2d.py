import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import Canvas
from typing import List

import numpy as np
import numpy.typing as npt

from geometry.bezier import BezierCurve
from geometry.two_d.box_2d import Box2d
from visualize.colors import COLORS, RGB


@dataclass
class VisualizationPoint2d:
    position_curve: npt.NDArray[np.float64]  # (N, dims)
    color: RGB

    @classmethod
    def from_ctrl_points(
        cls, ctrl_points: npt.NDArray[np.float64], color: str = "red1"
    ) -> "VisualizationPoint2d":
        position_curve = BezierCurve.create_from_ctrl_points(
            ctrl_points
        ).eval_entire_interval()
        return cls(position_curve, COLORS[color])


@dataclass
class VisualizationForce2d(VisualizationPoint2d):
    force_curve: npt.NDArray[np.float64]  # (N, dims)

    @classmethod
    def from_ctrl_points(
        cls,
        ctrl_points_position: npt.NDArray[
            np.float64
        ],  # TODO: create a struct or class for ctrl points
        ctrl_points_force: npt.NDArray[np.float64],
        color: str = "red1",
    ) -> "VisualizationForce2d":

        position_curve = BezierCurve.create_from_ctrl_points(
            ctrl_points_position
        ).eval_entire_interval()

        force_curve = BezierCurve.create_from_ctrl_points(
            ctrl_points_force
        ).eval_entire_interval()

        return cls(position_curve, COLORS[color], force_curve)


@dataclass
class VisualizationPolygon2d(VisualizationPoint2d):
    vertices_curves: List[npt.NDArray[np.float64]]  # [(N, dims), (N,dims), ...]

    @classmethod
    def from_ctrl_points(
        cls,
        ctrl_points_position: npt.NDArray[
            np.float64
        ],  # TODO: create a struct or class for ctrl points
        ctrl_points_orientation: List[npt.NDArray[np.float64]],
        box: Box2d,
        color: str = "red1",
    ) -> "VisualizationPolygon2d":

        position_curve = BezierCurve.create_from_ctrl_points(
            ctrl_points_position
        ).eval_entire_interval()

        temp = np.array(
            [
                pos.reshape((-1, 1)) + rot.dot(box.vertices_for_plotting)
                for rot, pos in zip(ctrl_points_orientation, ctrl_points_position.T)
            ]
        )  # (ctrl_points, dims, num_vertices)
        temp2 = np.transpose(temp, axes=[1, 0, 2])  # (dims, ctrl_points, num_vertices)
        num_vertices = temp2.shape[2]
        corner_ctrl_points_in_W = [temp2[:, :, idx] for idx in range(num_vertices)]

        corner_curves = [
            BezierCurve.create_from_ctrl_points(ctrl_points).eval_entire_interval()
            for ctrl_points in corner_ctrl_points_in_W
        ]

        return cls(position_curve, COLORS[color], corner_curves)


class Visualizer2d:
    WINDOW_WIDTH = 600
    WINDOW_HEIGHT = 450
    PLOT_CENTER = np.array([WINDOW_WIDTH / 2, WINDOW_HEIGHT * 3 / 4]).reshape((-1, 1))
    PLOT_SCALE = 500
    FORCE_SCALE = 0.02
    POINT_RADIUS = 0.01

    def visualize(
        self,
        points: List[VisualizationPoint2d],
        forces: List[VisualizationForce2d],
        polygons: List[VisualizationPolygon2d],
    ) -> None:
        curve_lengths = np.array(
            [len(p.position_curve) for p in points]
            + [len(f.position_curve) for f in forces]
        )
        if not np.all(curve_lengths == curve_lengths[0]):
            raise ValueError("Curve lengths for plotting must match!")
        num_frames = curve_lengths[0]

        self.app = tk.Tk()
        self.app.title("box")

        self.canvas = Canvas(
            self.app, width=self.WINDOW_WIDTH, height=self.WINDOW_HEIGHT, bg="white"
        )
        self.canvas.pack()

        for frame_idx in range(num_frames):
            self.canvas.delete("all")

            for polygon in polygons:
                self._draw_polygon(polygon, frame_idx)

            for point in points:
                self._draw_point(point, frame_idx)

            for force in forces:
                self._draw_force(force, frame_idx)

            self.canvas.update()
            time.sleep(0.05)
            if frame_idx == 0:
                time.sleep(2)

        self.app.mainloop()

    def _transform_points_for_plotting(
        self,
        points: npt.NDArray[np.float64],
    ) -> List[float]:
        points_flipped_y_axis = np.vstack([points[0, :], -points[1, :]])
        points_transformed = points_flipped_y_axis * self.PLOT_SCALE + self.PLOT_CENTER
        plotable_points = self._flatten_points(points_transformed)
        return plotable_points

    @staticmethod
    def _flatten_points(points: npt.NDArray[np.float64]) -> List[float]:
        return list(points.flatten(order="F"))

    def _draw_point(
        self, point: VisualizationPoint2d, idx: int, radius: float = POINT_RADIUS
    ) -> None:
        pos = point.position_curve[idx].reshape((-1, 1))

        lower_left = pos - radius
        upper_right = pos + radius
        points = self._transform_points_for_plotting(
            np.hstack([lower_left, upper_right])
        )
        self.canvas.create_oval(
            points[0],
            points[1],
            points[2],
            points[3],
            fill=point.color.hex_format(),
            width=0,
        )

    def _draw_force(self, force: VisualizationForce2d, idx: int) -> None:
        force_strength = force.force_curve[idx] * self.FORCE_SCALE
        force_endpoints = np.hstack(
            [
                force.position_curve[idx].reshape((-1, 1)),
                (force.position_curve[idx] + force_strength).reshape((-1, 1)),
            ]
        )
        force_points = self._transform_points_for_plotting(force_endpoints)
        self.canvas.create_line(
            force_points, width=2, arrow=tk.LAST, fill=force.color.hex_format()
        )

    def _draw_polygon(self, polygon: VisualizationPolygon2d, idx: int) -> None:
        vertices = np.hstack([c[idx].reshape((-1, 1)) for c in polygon.vertices_curves])
        polygon_points = self._transform_points_for_plotting(vertices)
        self.canvas.create_polygon(polygon_points, fill=polygon.color.hex_format())

        DARKENING = 50
        com_color = RGB(
            polygon.color.red - DARKENING,
            polygon.color.green - DARKENING,
            polygon.color.blue - DARKENING,
        )
        viz_com = VisualizationPoint2d(polygon.position_curve, com_color)
        self._draw_point(
            viz_com, idx, radius=self.POINT_RADIUS * 1.2
        )  # make com points a bit bigger
