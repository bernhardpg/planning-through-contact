import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import Canvas
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from planning_through_contact.geometry.bezier import BezierCurve
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.two_d.rigid_body_2d import RigidBody2d
from planning_through_contact.visualize.colors import COLORS, RGB


@dataclass
class VisualizationPoint2d:
    position_curve: npt.NDArray[np.float64]  # (N, dims)
    color: RGB

    # TODO remove
    @classmethod
    def from_ctrl_points(
        cls, ctrl_points: npt.NDArray[np.float64], color: str = "red1"
    ) -> "VisualizationPoint2d":
        position_curve = BezierCurve.create_from_ctrl_points(
            ctrl_points
        ).eval_entire_interval()
        return cls(position_curve, COLORS[color])

    def __post_init__(self):
        self.radius = 1.0

    def change_radius(self, new_radius: float) -> None:
        self.radius = new_radius


@dataclass
class VisualizationForce2d(VisualizationPoint2d):
    force_curve: npt.NDArray[np.float64]  # (N, dims)

    # TODO remove
    @classmethod
    def from_ctrl_points(
        cls,
        ctrl_points_position: npt.NDArray[np.float64],
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
    def from_trajs(
        cls,
        com_position: npt.NDArray[np.float64],  # (num_steps, n)
        rotation: npt.NDArray[np.float64],  # (num_steps, n ** 2)
        polytope: RigidBody2d
        | CollisionGeometry,  # TODO: RigidBody2d will be deprecated!
        color: RGB,
    ) -> "VisualizationPolygon2d":
        # Transform points into world frame
        # (some intermediate steps here to get the points in the right format)
        num_dims = 2  # TODO replace
        temp = np.array(
            [
                pos.reshape((-1, 1))
                + rot.reshape(num_dims, num_dims).dot(polytope.vertices_for_plotting)
                for rot, pos in zip(rotation, com_position)
            ]
        )  # (num_steps, dims, num_vertices)
        temp2 = np.transpose(temp, axes=[1, 0, 2])  # (dims, num_steps, num_vertices)
        num_vertices = temp2.shape[2]
        corner_curves = [temp2[:, :, idx].T for idx in range(num_vertices)]

        return cls(com_position, color, corner_curves)

    # TODO remove
    @classmethod
    def from_ctrl_points(
        cls,
        ctrl_points_com_position: npt.NDArray[np.float64],
        ctrl_points_orientation: List[npt.NDArray[np.float64]],
        polytope: RigidBody2d,
        color: str = "red1",
    ) -> "VisualizationPolygon2d":
        com_position_curve = BezierCurve.create_from_ctrl_points(
            ctrl_points_com_position
        ).eval_entire_interval()

        # Transform points into world frame
        # (some intermediate steps here to get the points in the right format)
        temp = np.array(
            [
                pos.reshape((-1, 1)) + rot.dot(polytope.vertices_for_plotting)
                for rot, pos in zip(ctrl_points_orientation, ctrl_points_com_position.T)
            ]
        )  # (ctrl_points, dims, num_vertices)
        temp2 = np.transpose(temp, axes=[1, 0, 2])  # (dims, ctrl_points, num_vertices)
        num_vertices = temp2.shape[2]
        corner_ctrl_points_in_W = [temp2[:, :, idx] for idx in range(num_vertices)]

        corner_curves = [
            BezierCurve.create_from_ctrl_points(ctrl_points).eval_entire_interval()
            for ctrl_points in corner_ctrl_points_in_W
        ]

        return cls(com_position_curve, COLORS[color], corner_curves)


class VisualizationCone2d(VisualizationPolygon2d):
    RAY_LENGTH: float = 0.05

    # TODO remove
    @classmethod
    def from_ctrl_points(
        cls,
        ctrl_points_position: npt.NDArray[np.float64],
        ctrl_points_orientation: List[npt.NDArray[np.float64]],
        normal_vec_ctrl_points: npt.NDArray[np.float64],
        angle: float,
        color: str = "cadetblue1",
    ) -> "VisualizationCone2d":
        position_curve = BezierCurve.create_from_ctrl_points(
            ctrl_points_position
        ).eval_entire_interval()

        normal_vec = normal_vec_ctrl_points[
            :, 0
        ]  # the normal vec is constant in the local frame. I have to clean up this code anyway
        base_angle = np.arctan2(normal_vec[1], normal_vec[0])
        ray_1 = (
            np.array([np.cos(base_angle - angle), np.sin(base_angle - angle)]).reshape(
                (-1, 1)
            )
            * cls.RAY_LENGTH
        )
        ray_2 = (
            np.array([np.cos(base_angle + angle), np.sin(base_angle + angle)]).reshape(
                (-1, 1)
            )
            * cls.RAY_LENGTH
        )
        rays = np.hstack((ray_1, ray_2))

        # Transform points into world frame
        # (some intermediate steps here to get the points in the right format)
        temp = np.array(
            [
                pos.reshape((-1, 1)) + rot.dot(rays)
                for rot, pos in zip(ctrl_points_orientation, ctrl_points_position.T)
            ]
        )  # (ctrl_points, dims, num_vertices)
        temp2 = np.transpose(temp, axes=[1, 0, 2])  # (dims, ctrl_points, num_vertices)
        num_rays = temp2.shape[2]
        corner_ctrl_points_in_W = [temp2[:, :, idx] for idx in range(num_rays)]

        ray_curves = [
            BezierCurve.create_from_ctrl_points(ctrl_points).eval_entire_interval()
            for ctrl_points in corner_ctrl_points_in_W
        ]
        vertices_curves = ray_curves + [position_curve]
        return cls(position_curve, COLORS[color], vertices_curves)


class Visualizer2d:
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 900
    PLOT_CENTER = np.array([WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2]).reshape((-1, 1))
    PLOT_SCALE = 500
    FORCE_SCALE = 0.4
    POINT_RADIUS = 0.01

    def visualize(
        self,
        points: List[VisualizationPoint2d],
        forces: List[VisualizationForce2d],
        polygons: List[VisualizationPolygon2d],
        frames_per_sec: float = 20.0,
        target: Optional[VisualizationPolygon2d] = None,
        draw_origin: bool = False,
    ) -> None:
        curve_lengths = np.array(
            [len(p.position_curve) for p in points]
            + [len(f.position_curve) for f in forces]
        )
        if not np.all(curve_lengths == curve_lengths[0]):
            raise ValueError("Curve lengths for plotting must match!")
        num_frames = curve_lengths[0]
        pause_between_frames = 1 / frames_per_sec

        self.root = tk.Tk()
        self.root.title("Visualization")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # calculate x and y coordinates for the Tk root window
        x = (screen_width / 2) - (self.WINDOW_WIDTH / 2)
        y = (screen_height / 2) - (self.WINDOW_HEIGHT / 2)

        # Set the dimensions of the screen
        # and where it is placed
        self.root.geometry(
            "%dx%d+%d+%d" % (self.WINDOW_WIDTH, self.WINDOW_HEIGHT, x, y)
        )

        self.canvas = Canvas(
            self.root, width=self.WINDOW_WIDTH, height=self.WINDOW_HEIGHT, bg="white"
        )
        self.canvas.pack()

        if draw_origin:
            origin = VisualizationPoint2d(np.zeros((1, 2)), COLORS["cornsilk4"])
            origin.change_radius(4.0)

        for frame_idx in range(num_frames):
            self.canvas.delete("all")

            if target is not None:
                self._draw_target_polygon(target)

            for polygon in polygons:
                self._draw_polygon(polygon, frame_idx)

            for point in points:
                self._draw_point(point, frame_idx)

            if draw_origin:
                self._draw_point(origin, 0)  # type: ignore

            for force in forces:
                self._draw_force(force, frame_idx)

            self.canvas.update()
            time.sleep(pause_between_frames)
            if frame_idx == 0:
                time.sleep(2)

        self.root.mainloop()

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

    def _draw_point(self, point: VisualizationPoint2d, idx: int) -> None:
        radius = self.POINT_RADIUS * point.radius
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
        ARROW_HEAD_LENGTH = 0.01
        if np.linalg.norm(force_strength) < ARROW_HEAD_LENGTH:
            self.canvas.create_line(
                force_points, width=2, fill=force.color.hex_format()
            )
        else:
            self.canvas.create_line(
                force_points, width=2, arrow=tk.LAST, fill=force.color.hex_format()
            )

    def _draw_polygon(
        self, polygon: VisualizationPolygon2d, idx: int, plot_com: bool = False
    ) -> None:
        vertices = np.hstack([c[idx].reshape((-1, 1)) for c in polygon.vertices_curves])
        polygon_points = self._transform_points_for_plotting(vertices)
        self.canvas.create_polygon(
            polygon_points, fill=polygon.color.hex_format(), stipple="gray50"
        )
        if plot_com:
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

    def _draw_target_polygon(self, target: VisualizationPolygon2d) -> None:
        LAST_IDX = -1  # we always use the last idx for target
        vertices = np.hstack(
            [c[LAST_IDX].reshape((-1, 1)) for c in target.vertices_curves]
        )
        polygon_points = self._transform_points_for_plotting(vertices)
        self.canvas.create_polygon(
            polygon_points, outline=target.color.hex_format(), dash=(10, 5), width=3
        )
