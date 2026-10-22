"""Tests for what gets installed: the built wheel and sdist, the version, and settings files.

The builds run in-process with hatchling against a copy of the repository, so nothing is
downloaded and nothing is written into the working tree. The copy has private-looking
files planted in it, because the sdist is published and a working tree for this project
is exactly where an untracked chat export ends up.
"""

from __future__ import annotations

import email
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

from confidant import __version__
from confidant.cli import main
from confidant.config import Settings, user_env_file

ROOT = Path(__file__).parent.parent

# Planted in the copy before building; none of them may reach the sdist.
PLANTED = (".env", "notes.docx", "data/chat.txt", "transcripts/robin.txt", "me.db")
SDIST_ALLOWED = {
    "src",
    "tests",
    "examples",
    "docs",
    "README.md",
    "ROADMAP.md",
    "CHANGELOG.md",
    "LICENSE",
    ".env.example",
    # Always added by hatchling.
    "pyproject.toml",
    "PKG-INFO",
    ".gitignore",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    hatchling = pytest.importorskip("hatchling.builders.wheel")
    from hatchling.builders.sdist import SdistBuilder

    tree = tmp_path_factory.mktemp("tree") / "confidant"
    # The real .env and store are left out of the copy, not just out of the build: there
    # is no reason for a test to duplicate the developer's secrets into /tmp.
    shutil.copytree(
        ROOT,
        tree,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".env", "*.db"
        ),
    )
    for name in PLANTED:
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text("private\n")

    out = tmp_path_factory.mktemp("dist")
    [wheel] = hatchling.WheelBuilder(str(tree)).build(directory=str(out), versions=["standard"])
    [sdist] = SdistBuilder(str(tree)).build(directory=str(out), versions=["standard"])
    return Path(wheel), Path(sdist)


def _wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def _wheel_text(wheel: Path, suffix: str) -> str:
    with zipfile.ZipFile(wheel) as archive:
        [name] = [n for n in archive.namelist() if n.endswith(suffix)]
        return archive.read(name).decode()


def test_wheel_is_named_for_pypi_and_carries_the_package_version(built):
    wheel, sdist = built
    assert wheel.name.startswith(f"confidant_agent-{__version__}-")
    assert sdist.name == f"confidant_agent-{__version__}.tar.gz"
    metadata = email.message_from_string(_wheel_text(wheel, ".dist-info/METADATA"))
    assert metadata["Name"] == "confidant-agent"
    assert metadata["Version"] == __version__
    assert "api" in metadata.get_all("Provides-Extra")


def test_serve_dependencies_stay_behind_the_api_extra(built):
    wheel, _ = built
    metadata = email.message_from_string(_wheel_text(wheel, ".dist-info/METADATA"))
    requires = metadata.get_all("Requires-Dist")
    for name in ("fastapi", "uvicorn"):
        # The dev extra pulls in the api one, so each appears once per extra, never bare.
        lines = [r for r in requires if r.startswith(name)]
        assert lines
        assert all("extra ==" in line for line in lines)
        assert any(re.search(r"extra == ['\"]api['\"]", line) for line in lines)


def test_wheel_installs_the_confidant_command(built):
    wheel, _ = built
    entry_points = _wheel_text(wheel, ".dist-info/entry_points.txt")
    assert re.search(r"^confidant = confidant\.cli:main$", entry_points, re.MULTILINE)


def test_wheel_ships_the_web_ui_and_every_module(built):
    wheel, _ = built
    names = set(_wheel_names(wheel))
    for asset in ("index.html", "app.js", "style.css"):
        assert f"confidant/web/{asset}" in names
    source = ROOT / "src"
    for module in source.rglob("*.py"):
        assert module.relative_to(source).as_posix() in names
    # Only the package itself and its metadata: no tests, examples, or stray files.
    outside = {n for n in names if not n.startswith(("confidant/", "confidant_agent-"))}
    assert outside == set()


def test_sdist_holds_only_the_allowlist(built):
    _, sdist = built
    with tarfile.open(sdist) as archive:
        # Members are prefixed with "confidant_agent-<version>/".
        tops = {Path(m.name).parts[1] for m in archive.getmembers() if "/" in m.name}
        names = {Path(*Path(m.name).parts[1:]).as_posix() for m in archive.getmembers()}
    assert tops <= SDIST_ALLOWED
    for planted in PLANTED:
        assert planted not in names
    assert not [n for n in names if n.endswith((".db", ".docx", ".pdf"))]
    # The fictional examples and the env template are meant to travel with it.
    assert "examples/sample_chat.txt" in names
    assert ".env.example" in names


def test_version_flag_reports_the_package_version(capsys):
    with pytest.raises(SystemExit) as exit_:
        main(["--version"])
    assert exit_.value.code == 0
    assert capsys.readouterr().out.strip() == f"confidant {__version__}"


def test_changelog_has_a_section_for_this_version():
    # The release workflow takes the release notes from this section.
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert f"## [{__version__}]" in changelog


# --- Settings files -------------------------------------------------------------------


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """A home and a working directory of the test's own, and an environment to discard.

    load_dotenv writes into os.environ, so the test gets a copy of it: monkeypatch.delenv
    on a variable that is not set yet records nothing to restore.
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for var in ("ANTHROPIC_API_KEY", "CONFIDANT_MODEL", "CONFIDANT_EFFORT"):
        os.environ.pop(var, None)
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    return home, work


def test_user_env_file_sits_beside_the_store(isolated):
    home, _ = isolated
    assert user_env_file() == home / ".confidant" / ".env"


def test_home_env_file_is_read_from_any_directory(isolated):
    home, _ = isolated
    (home / ".confidant").mkdir()
    (home / ".confidant" / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-home\n")
    assert Settings.from_env().api_key == "sk-ant-from-home"


def test_working_directory_env_wins_over_the_home_file(isolated):
    home, work = isolated
    (home / ".confidant").mkdir()
    (home / ".confidant" / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-ant-from-home\nCONFIDANT_EFFORT=low\n"
    )
    (work / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-here\n")
    settings = Settings.from_env()
    assert settings.api_key == "sk-ant-from-here"
    # Files merge: what the nearer one does not set still comes from home.
    assert settings.effort == "low"


def test_env_file_above_the_working_directory_is_found(isolated):
    _, work = isolated
    (work / ".env").write_text("CONFIDANT_MODEL=claude-from-parent\n")
    nested = work / "a" / "b"
    nested.mkdir(parents=True)
    os.chdir(nested)
    assert Settings.from_env().model == "claude-from-parent"


def test_the_shell_wins_over_both_files(isolated, monkeypatch):
    home, work = isolated
    (home / ".confidant").mkdir()
    (home / ".confidant" / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-home\n")
    (work / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-here\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    assert Settings.from_env().api_key == "sk-ant-from-shell"


def test_no_env_files_is_fine(isolated):
    assert Settings.from_env().api_key is None
