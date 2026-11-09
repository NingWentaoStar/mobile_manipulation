"""What a skill is: a named action with a stated scope, returning a SkillResult."""
from dataclasses import dataclass, field


@dataclass
class SkillResult:
    success: bool
    message: str = ""
    info: dict = field(default_factory=dict)

    def __bool__(self):
        return self.success


class SkillError(RuntimeError):
    def __init__(self, message, info=None):
        super().__init__(message)
        self.info = info or {}


class Skill:
    """Subclass, set the class attributes, write execute(**params) and decorate with @register. execute() returns
    a dict of details and raises SkillError on failure."""
    name = ""
    applies_to = ""
    not_for = ""
    preconditions = ""

    def __init__(self, context):
        self.context = context

    def label(self, **params):
        return self.name

    def execute(self, **params):
        raise NotImplementedError

    def run(self, **params):
        start, label = self.context.steps, self.label(**params)
        try:
            info = self.execute(**params) or {}
            return SkillResult(True, f"{label} done", dict(info, steps=self.context.steps - start))
        except SkillError as error:
            return SkillResult(False, f"{label} failed: {error}", dict(error.info, steps=self.context.steps - start))
