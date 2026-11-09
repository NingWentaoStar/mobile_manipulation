"""skills/catalog.json matches the code and the files it names."""
import json

from skills.paths import REPO

SKILLS = json.loads((REPO / "skills/catalog.json").read_text(encoding="utf-8"))["local_skills"]
REQUIRED = {"id", "name", "kind", "call", "code", "demo", "docs", "preconditions", "applies_to", "not_for", "validated"}


def test_every_entry_is_complete():
    assert len({s["id"] for s in SKILLS}) == len(SKILLS)
    for skill in SKILLS:
        assert REQUIRED <= set(skill), skill["id"]
        assert skill["kind"] in ("classical", "rl")
        assert skill["validated"]["results"], skill["id"]


def test_named_files_exist():
    for skill in SKILLS:
        for key in ("code", "demo", "docs", "checkpoint", "training"):
            if key in skill:
                assert (REPO / skill[key]).is_file(), (skill["id"], key)


def test_each_entry_names_a_skill_in_its_module():
    for skill in SKILLS:
        assert f'name = "{skill["id"]}"' in (REPO / skill["code"]).read_text(encoding="utf-8"), skill["id"]
        assert f'skills.run("{skill["id"]}"' in skill["call"], skill["id"]


def test_every_skill_module_is_listed():
    listed = {skill["code"] for skill in SKILLS}
    folder = REPO / "skills/library"
    for path in [*folder.glob("*.py"), *(folder / "learned").glob("*.py")]:
        if "\n@register\n" in path.read_text(encoding="utf-8"):
            assert path.relative_to(REPO).as_posix() in listed, path.name
