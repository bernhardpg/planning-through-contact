import numpy as np
from pydrake.solvers import CommonSolverOption, MosekSolver, SolverOptions

from planning_through_contact.experiments.deprecated.planar_pushing.old.planar_pushing_gcs import (
    DynamicsConfig,
    PlanarPushingContactMode,
)
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


# TODO(bernhardpg): This can safely be deleted once I delete planar_pushing_gcs.py
def test_temp_test_old_mode() -> None:
    """
    Test whether the old contact mode formulation is as loose as the new one
    (it seems it is)
    (also this is not really a unit test)
    """
    object = RigidBody("tee", TPusher2d(), mass=1)
    VIS_REALTIME_RATE = 0.25

    num_knot_points = 4
    contact_face_idx = 3

    initial_pose = PlanarPose(0, 0, 0)
    final_pose = PlanarPose(0.3, 0, 0.1)

    VIS_REALTIME_RATE = 1.0
    time_in_contact = 2

    f_max = 0.5 * 9.81 * object.mass
    dynamics_config = DynamicsConfig(
        friction_coeff_table_slider=0.5,
        friction_coeff_slider_pusher=0.5,
        f_max=f_max,
        tau_max=f_max * 0.2,
    )

    mode = PlanarPushingContactMode(
        object,
        contact_face_idx,
        dynamics_config,
        num_knot_points,
        time_in_contact,
        initial_pose.theta,
        final_pose.theta,
        initial_pose.pos(),
        final_pose.pos(),
    )

    solver_options = SolverOptions()

    DEBUG = False
    if DEBUG:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    result = MosekSolver().Solve(mode.relaxed_prog, solver_options=solver_options)  # type: ignore
    assert result.is_success()  # should fail when the relaxation is tight!

    vals = [mode.eval_result(result)]

    interpolate = False
    DT = 0.5
    R_traj = sum(
        [val.get_R_traj(DT, interpolate=interpolate) for val in vals],
        [],
    )

    for R in R_traj:
        np.abs(np.linalg.det(R))
        # relaxation will not be tight
        # assert det <= 1 + eps and det >= 1 - eps

    com_traj = np.vstack(
        [val.get_p_WB_traj(DT, interpolate=interpolate) for val in vals]
    )
    force_traj = np.vstack(
        [val.get_f_c_W_traj(DT, interpolate=interpolate) for val in vals]
    )
    contact_pos_traj = np.vstack(
        [val.get_p_c_W_traj(DT, interpolate=interpolate) for val in vals]
    )
    len(R_traj)

    CONTACT_COLOR = COLORS["dodgerblue4"]
    GRAVITY_COLOR = COLORS["blueviolet"]
    BOX_COLOR = COLORS["aquamarine4"]
    COLORS["bisque3"]
    FINGER_COLOR = COLORS["firebrick3"]
    TARGET_COLOR = COLORS["firebrick1"]
    COLORS["darkorange1"]

    if DEBUG:
        flattened_rotation = np.vstack([R.flatten() for R in R_traj])
        box_viz = VisualizationPolygon2d.from_trajs(
            com_traj,
            flattened_rotation,
            object,
            BOX_COLOR,
        )

        # NOTE: I don't really need the entire trajectory here, but leave for now
        target_viz = VisualizationPolygon2d.from_trajs(
            com_traj,
            flattened_rotation,
            object,
            TARGET_COLOR,
        )

        com_points_viz = VisualizationPoint2d(com_traj, GRAVITY_COLOR)  # type: ignore
        contact_point_viz = VisualizationPoint2d(contact_pos_traj, FINGER_COLOR)  # type: ignore
        contact_force_viz = VisualizationForce2d(contact_pos_traj, CONTACT_COLOR, force_traj)  # type: ignore

        # visualize velocity with an arrow (i.e. as a force), and reverse force scaling

        viz = Visualizer2d()
        FRAMES_PER_SEC = len(R_traj) / (time_in_contact / VIS_REALTIME_RATE)
        viz.visualize(
            [contact_point_viz],
            [contact_force_viz],
            [box_viz],
            FRAMES_PER_SEC,
            target_viz,
        )
