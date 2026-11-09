"""Rigid transform with no simulator dependency: p (xyz), q (wxyz)."""
import numpy as np


def quat_from_matrix(m):
    """Shepperd's method: divides by the largest component, so rotations near 180 deg stay exact."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    i = int(np.argmax([trace, m[0, 0], m[1, 1], m[2, 2]]))
    if i == 0:
        q = np.array([1 + trace, m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    elif i == 1:
        q = np.array([m[2, 1] - m[1, 2], 1 + 2*m[0, 0] - trace, m[0, 1] + m[1, 0], m[0, 2] + m[2, 0]])
    elif i == 2:
        q = np.array([m[0, 2] - m[2, 0], m[0, 1] + m[1, 0], 1 + 2*m[1, 1] - trace, m[1, 2] + m[2, 1]])
    else:
        q = np.array([m[1, 0] - m[0, 1], m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], 1 + 2*m[2, 2] - trace])
    return q / np.linalg.norm(q)


def matrix_from_quat(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])


class Pose:
    def __init__(self, p=(0, 0, 0), q=(1, 0, 0, 0)):
        p = np.asarray(p, dtype=np.float64)
        if p.shape == (4, 4):
            self.p, self.q = p[:3, 3].copy(), quat_from_matrix(p[:3, :3])
        else:
            self.p, self.q = p.copy(), np.asarray(q, dtype=np.float64) / np.linalg.norm(q)

    def to_transformation_matrix(self):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = matrix_from_quat(self.q), self.p
        return m

    def __mul__(self, other):
        return Pose(self.to_transformation_matrix() @ other.to_transformation_matrix())

    def inv(self):
        return Pose(np.linalg.inv(self.to_transformation_matrix()))

    def __repr__(self):
        return f"Pose({self.p.tolist()}, {self.q.tolist()})"


def pose_from_axes(position, approach, closing):
    """TCP pose from the gripper approach direction (TCP +z) and finger closing direction (TCP +y)."""
    z = np.asarray(approach, float); z /= np.linalg.norm(z)
    y = np.asarray(closing, float); y = y - z * (y @ z); y /= np.linalg.norm(y)
    x = np.cross(y, z)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, position
    return Pose(m)
