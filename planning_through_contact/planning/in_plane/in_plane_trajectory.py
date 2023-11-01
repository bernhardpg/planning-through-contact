from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import numpy.typing as npt
from pydrake.solvers import MathematicalProgramResult

from planning_through_contact.geometry.in_plane.contact_scene import (
    ContactSceneCtrlPoint,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.in_plane.contact_scene_program import (
    ContactSceneProgram,
)
from planning_through_contact.tools.utils import evaluate_np_expressions_array


def _get_positions_for_body_for_all_ctrl_points(
    body: RigidBody,
    ctrl_points: List[ContactSceneCtrlPoint],
    result: MathematicalProgramResult,
) -> List[npt.NDArray[np.float64]]:
    body_pos_symbolic = [
        ctrl_point.get_body_pos_in_world(body) for ctrl_point in ctrl_points
    ]
    body_pos = [
        evaluate_np_expressions_array(expr, result) for expr in body_pos_symbolic
    ]
    return body_pos


def _get_rotations_for_body_for_all_ctrl_points(
    body: RigidBody,
    ctrl_points: List[ContactSceneCtrlPoint],
    result: MathematicalProgramResult,
) -> List[npt.NDArray[np.float64]]:
    body_rot_symbolic = [
        ctrl_point.get_body_rot_in_world(body) for ctrl_point in ctrl_points
    ]
    body_rot = [
        evaluate_np_expressions_array(expr, result) for expr in body_rot_symbolic
    ]
    return body_rot


@dataclass
class InPlaneTrajectory:
    bodies: List[RigidBody]
    body_positions: Dict[RigidBody, List[npt.NDArray[np.float64]]]
    body_rotations: Dict[RigidBody, List[npt.NDArray[np.float64]]]

    @classmethod
    def create_from_result(
        cls, result: MathematicalProgramResult, problem: ContactSceneProgram
    ) -> "InPlaneTrajectory":
        bodies = problem.contact_scene_def.rigid_bodies
        body_positions = {
            body: _get_positions_for_body_for_all_ctrl_points(
                body, problem.ctrl_points, result
            )
            for body in bodies
        }
        body_rotations = {
            body: _get_rotations_for_body_for_all_ctrl_points(
                body, problem.ctrl_points, result
            )
            for body in bodies
        }

        return cls(bodies, body_positions, body_rotations)

    # NOTE: Only used for legacy plotter
    def get_flat_body_rotations(self, body: RigidBody) -> List[npt.NDArray[np.float64]]:
        return [
            rot.flatten().reshape((-1, 1)) for rot in self.body_rotations[body]
        ]  # (4,1)
