"""Pose math (no simulator needed)."""
import numpy as np

from skills.manipulation.pose import Pose, matrix_from_quat, pose_from_axes, quat_from_matrix


def random_rotation(rng):
    q = rng.normal(size=4)
    return matrix_from_quat(q / np.linalg.norm(q))


def test_matrix_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(500):
        m = random_rotation(rng)
        assert np.allclose(matrix_from_quat(quat_from_matrix(m)), m, atol=1e-9)


def test_half_turns_keep_their_axis():
    rng = np.random.default_rng(1)
    for _ in range(200):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        m = 2*np.outer(axis, axis) - np.eye(3)
        assert np.allclose(matrix_from_quat(quat_from_matrix(m)), m, atol=1e-9)


def test_grasp_axes_survive():
    for approach, closing in (((0, 1, 0), (0, 0, 1)), ((0, 0, -1), (1, 0, 0)), ((0, 0, -1), (-0.6, 0.8, 0)), ((1, 0, 0), (0, 0, -1))):
        m = pose_from_axes(np.zeros(3), approach, closing).to_transformation_matrix()
        assert np.allclose(m[:3, 2], approach) and np.allclose(m[:3, 1], closing)


def test_compose_and_invert():
    rng = np.random.default_rng(2)
    a, b = Pose(rng.normal(size=3), rng.normal(size=4)), Pose(rng.normal(size=3), rng.normal(size=4))
    assert np.allclose((a * b).to_transformation_matrix(), a.to_transformation_matrix() @ b.to_transformation_matrix())
    assert np.allclose((a * a.inv()).to_transformation_matrix(), np.eye(4), atol=1e-9)
