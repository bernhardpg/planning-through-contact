from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import numpy.typing as npt
from pydrake.solvers import MathematicalProgramResult

from planning_through_contact.geometry.in_plane.contact_force import ContactForce
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


def _get_contact_force_for_idx(
    idx: int,
    ctrl_points: List[ContactSceneCtrlPoint],
    result: MathematicalProgramResult,
) -> List[npt.NDArray[np.float64]]:
    contact_force_symbolic = [
        ctrl_point.get_contact_forces_in_world_frame()[idx]
        for ctrl_point in ctrl_points
    ]
    contact_force = [
        evaluate_np_expressions_array(expr, result) for expr in contact_force_symbolic
    ]
    return contact_force


def _get_contact_point_for_idx(
    idx: int,
    ctrl_points: List[ContactSceneCtrlPoint],
    result: MathematicalProgramResult,
) -> List[npt.NDArray[np.float64]]:
    contact_point_symbolic = [
        ctrl_point.get_contact_positions_in_world_frame()[idx]
        for ctrl_point in ctrl_points
    ]
    contact_point = [
        evaluate_np_expressions_array(expr, result) for expr in contact_point_symbolic
    ]
    return contact_point


@dataclass
class InPlaneTrajectory:
    bodies: List[RigidBody]
    body_positions: Dict[RigidBody, List[npt.NDArray[np.float64]]]
    body_rotations: Dict[RigidBody, List[npt.NDArray[np.float64]]]
    contact_forces: Dict[str, List[npt.NDArray[np.float64]]]
    contact_positions: Dict[str, List[npt.NDArray[np.float64]]]

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

        # TODO(bernhardpg): We want to associate each force with the body it acts on
        num_contact_forces = len(
            problem.ctrl_points[0].get_contact_forces_in_world_frame()
        )
        contact_forces = {
            f"force_{idx}": _get_contact_force_for_idx(idx, problem.ctrl_points, result)
            for idx in range(num_contact_forces)
        }
        contact_positions = {
            f"force_{idx}": _get_contact_point_for_idx(idx, problem.ctrl_points, result)
            for idx in range(num_contact_forces)
        }  # one for each force

        return cls(
            bodies, body_positions, body_rotations, contact_forces, contact_positions
        )

    # NOTE: Only used for legacy plotter
    def get_flat_body_rotations(self, body: RigidBody) -> List[npt.NDArray[np.float64]]:
        return [
            rot.flatten().reshape((-1, 1)) for rot in self.body_rotations[body]
        ]  # (4,1)
