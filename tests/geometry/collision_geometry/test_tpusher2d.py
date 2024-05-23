import numpy as np

from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d


def test_get_signed_distance() -> None:
    tee = TPusher2d()
    com = tee.com_offset.flatten()

    def _correct_sdf(pos, target_idx):
        """
        We specify pos relative to box_1 as that is simpler.
        """
        abs_pos = pos - com
        return np.isclose(
            tee.get_signed_distance(abs_pos), tee.faces[target_idx].dist_to(abs_pos)
        )

    pos = np.array([-tee.width / 2, 0])
    assert _correct_sdf(pos, 7)

    pos = np.array([-0.5, 0])
    assert _correct_sdf(pos, 7)

    pos = np.array([0.5, 0])
    assert _correct_sdf(pos, 1)

    pos = np.array([0, 0.5])
    assert _correct_sdf(pos, 0)

    pos = np.array([0, -0.5])
    assert _correct_sdf(pos, 4)

    pos = np.array([-0.5, -0.5])
    assert _correct_sdf(pos, 5)

    pos = np.array([-0.1, -0.03])
    assert _correct_sdf(pos, 6)

    pos = np.array([0.5, -0.5])
    assert _correct_sdf(pos, 3)

    pos = np.array([0.1, -0.03])
    assert _correct_sdf(pos, 2)


def test_get_jacobian() -> None:
    tee = TPusher2d()

    force_comps = np.array([0.1, 0.05]).reshape((2, 1))

    def _get_f(x, y):
        """
        We specify pos relative to box_1 as that is simpler.
        """
        pos_rel_to_com_box_1 = np.array([x, y]) - tee.com_offset.flatten()
        J = tee.get_contact_jacobian(np.array(pos_rel_to_com_box_1))
        f = J.T @ force_comps
        return f.flatten()

    # left
    f = _get_f(-tee.width / 2, 0)
    assert np.isclose(f[0], force_comps[0])
    assert np.isclose(f[1], force_comps[1])
    assert f[2] <= 0

    # right
    f = _get_f(tee.width / 2, 0)
    assert np.isclose(f[0], -force_comps[0])
    assert np.isclose(f[1], -force_comps[1])
    assert f[2] <= 0

    # top
    f = _get_f(0, 0.3)
    assert np.isclose(f[0], force_comps[1])
    assert np.isclose(f[1], -force_comps[0])
    assert f[2] <= 0

    # bottom
    f = _get_f(0, -0.3)
    assert np.isclose(f[0], -force_comps[1])
    assert np.isclose(f[1], force_comps[0])
    assert f[2] <= 0
