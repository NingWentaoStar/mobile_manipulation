"""The skill library: find a skill by name and run it."""
from .context import SkillContext

SKILLS = {}


def register(cls):
    SKILLS[cls.name] = cls
    return cls


class SkillLibrary:
    def __init__(self, scene, on_step=None, on_idle=None):
        self.scene = scene
        self.context = SkillContext(scene, on_step, on_idle)
        self.skills = {name: cls(self.context) for name, cls in SKILLS.items()}

    def reset(self, start=None):
        """Reset the scene, and the gripper and held object with it."""
        self.scene.reset(start)
        self.context.reset()

    def run(self, name, **params):
        if name not in self.skills:
            raise KeyError(f"unknown skill {name!r}; available: {sorted(self.skills)}")
        return self.skills[name].run(**params)

    def list(self):
        return {name: dict(applies_to=skill.applies_to, not_for=skill.not_for, preconditions=skill.preconditions)
                for name, skill in self.skills.items()}
