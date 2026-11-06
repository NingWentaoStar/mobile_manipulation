"""Command validation (no simulator needed)."""
import pytest

from mobile_panda_sim.ros2.protocol import validate_command


def test_valid_commands():
    validate_command({"kind": "velocity", "linear": 0.25, "angular": -0.7})
    validate_command({"kind": "arm", "delta": [0.1, -0.1, 0, 0, 0, 0, 0]})
    validate_command({"kind": "gripper", "width": 0.04})
    validate_command({"kind": "stop"})


@pytest.mark.parametrize("command", [
    {"kind": "velocity", "linear": 0.3},
    {"kind": "velocity", "angular": float("nan")},
    {"kind": "arm", "delta": [0.0] * 6},
    {"kind": "arm", "delta": [0.2, 0, 0, 0, 0, 0, 0]},
    {"kind": "gripper", "width": 0.05},
    {"kind": "gripper", "width": "open"},
    {"kind": "task"},
    ["not", "a", "dict"],
])
def test_rejected_commands(command):
    with pytest.raises(ValueError):
        validate_command(command)
