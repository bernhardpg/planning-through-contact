import numpy as np
import numpy.typing as npt
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.all import InverseKinematics
from pydrake.multibody.plant import MultibodyPlant
from pydrake.solvers import Solve


def solve_ik(
    plant: MultibodyPlant,
    pose: RigidTransform,
    default_joint_positions: npt.NDArray[np.float64],
    disregard_angle: bool = False,
) -> npt.NDArray[np.float64]:
    # Plant needs to be just the robot without other objects
    # Need to create a new context that the IK can use for solving the problem

    ik = InverseKinematics(plant, with_joint_limits=True)  # type: ignore
    pusher_frame = plant.GetFrameByName("pusher_end")
    EPS = 1e-3

    ik.AddPositionConstraint(
        pusher_frame,
        np.zeros(3),
        plant.world_frame(),
        pose.translation() - np.ones(3) * EPS,
        pose.translation() + np.ones(3) * EPS,
    )

    if disregard_angle:
        z_unit_vec = np.array([0, 0, 1])
        ik.AddAngleBetweenVectorsConstraint(
            pusher_frame,
            z_unit_vec,
            plant.world_frame(),
            -z_unit_vec,  # The pusher object has z-axis pointing up
            0 - EPS,
            0 + EPS,
        )

    else:
        ik.AddOrientationConstraint(
            pusher_frame,
            RotationMatrix(),
            plant.world_frame(),
            pose.rotation(),
            EPS,
        )

    # Cost on deviation from default joint positions
    prog = ik.get_mutable_prog()
    q = ik.q()

    q0 = default_joint_positions
    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
    prog.SetInitialGuess(q, q0)

    result = Solve(ik.prog())
    assert result.is_success()

    q_sol = result.GetSolution(q)
    return q_sol
