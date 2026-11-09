"""Skill library. Each skill is one class in its own module; importing them here registers them."""
from . import articulate, base_pose, navigate, pick, place, reset_arm  # noqa: F401
from .base import Skill, SkillError, SkillResult  # noqa: F401
from .learned import pick_move, place as learned_place  # noqa: F401
from .registry import SkillLibrary, register  # noqa: F401
