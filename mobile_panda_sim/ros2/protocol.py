"""Dependency-free command validation shared by the simulation worker, the ROS node and tests."""
import math

from ..geometry import GRIPPER_RANGE, MAX_ANGULAR, MAX_JOINT_DELTA, MAX_LINEAR

KINDS = {"velocity", "arm", "gripper", "stop", "reset", "shutdown"}


def finite(value, limit, name):
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(x) or abs(x) > limit + 1e-12:
        raise ValueError(f"{name} must be finite and within [-{limit}, {limit}]")
    return x


def validate_command(command):
    if not isinstance(command, dict):
        raise ValueError("command must be a JSON object")
    kind = command.get("kind")
    if kind not in KINDS:
        raise ValueError("unknown command kind")
    if kind == "velocity":
        finite(command.get("linear", 0), MAX_LINEAR, "linear.x (m/s)")
        finite(command.get("angular", 0), MAX_ANGULAR, "angular.z (rad/s)")
    elif kind == "arm":
        delta = command.get("delta")
        if not isinstance(delta, (list, tuple)) or len(delta) != 7:
            raise ValueError("arm_joint_delta needs 7 values (panda_joint1..7, rad)")
        for i, value in enumerate(delta):
            finite(value, MAX_JOINT_DELTA, f"delta[{i}] (rad)")
    elif kind == "gripper":
        width = command.get("width")
        try:
            width = float(width)
        except (TypeError, ValueError):
            raise ValueError("gripper target must be a number (m per finger)")
        low, high = GRIPPER_RANGE
        if not math.isfinite(width) or not low - 1e-12 <= width <= high + 1e-12:
            raise ValueError(f"gripper target must be within [{low}, {high}] m per finger")
    return command
