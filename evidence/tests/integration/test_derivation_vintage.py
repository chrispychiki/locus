import locus.evidence.derive as derive_mod
import pytest
from _support import FIXTURE, read_recording
from locus.evidence.db import connect
from locus.evidence.derive import VintageUnreadable
from locus.evidence.hydrate import hydrate

DEFINITION = {
    "session": {"inactivity_minutes": 30},
    "engaged": {"min_seconds": 10, "min_pageviews": 2},
    "path": "wherever",
}

PACKAGE = {
    "__init__.py": "",
    "hydrate.py": (
        'def load(event):\n    raise ValueError("no such event")\n\n\n'
        "def later():\n    pass\n"
    ),
    "replay_seek.js": "window.seek = (at) => at;\n",
}

DISTILL = {
    "package.json": '{"dependencies": {"aria-query": "5.3.2", "rrweb": "2.1.7"}}\n',
    "bun.lock": "the lock\n",
    "bunfig.toml": "[install]\n",
    "distill.js": 'import { roles } from "aria-query";\nexport const logic = roles;\n',
    "node_modules/aria-query/package.json": '{"name": "aria-query", "version": "5.3.2"}\n',
    "node_modules/rrweb/package.json": '{"name": "rrweb", "version": "2.1.7"}\n',
}


def _write(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.fixture
def code(tmp_path, monkeypatch):
    """A package and a distiller shaped like the real ones, standing in for them as the code the identity covers."""
    package, distill = tmp_path / "evidence", tmp_path / "distill"
    _write(package, PACKAGE)
    _write(distill, DISTILL)
    monkeypatch.setattr(derive_mod, "_code_roots", lambda: (package, distill))
    return package, distill


def code_vintage():
    return derive_mod._code_vintage()


def test_the_code_vintage_moves_with_the_code_and_nothing_else(code):
    # Derived from the code itself, never a hand-bumped constant: a forgotten bump would make the
    # detector affirm currency exactly when it lies.
    package, distill = code
    first = code_vintage()
    assert first == code_vintage(), "stable across calls"

    # A comment, a docstring or a reformat changes nothing derived — a function whose whole body
    # becomes its docstring included.
    _write(
        package,
        {
            "hydrate.py": '"""The canonical timestamp."""\n\n\ndef load(event):\n'
            '    """Read one event."""\n    raise ValueError("no such event")\n\n\n'
            'def later():\n    """Nothing yet."""\n',
            "replay_seek.js": "// seek the replayer\nwindow.seek  =  (at)=>at\n",
        },
    )
    _write(distill, {"distill.js": DISTILL["distill.js"] + "// how distilling works\n"})
    assert code_vintage() == first, "no code changed"

    # Every Python and JavaScript file of the package and the distiller counts, whatever reaches
    # it, and an error message is code like any other.
    for root, name, text in [
        (package, "hydrate.py", PACKAGE["hydrate.py"].replace("no such", "unknown")),
        (package, "anywhere.py", "LATER = 1\n"),
        (package, "replay_seek.js", "window.seek = (at) => at + 1;\n"),
        (distill, "distill.js", DISTILL["distill.js"] + "export const more = 1;\n"),
        (distill, "elsewhere.js", "export const later = 1;\n"),
    ]:
        before = code_vintage()
        _write(root, {name: text})
        assert code_vintage() != before, name

    # What is not our source counts for nothing: the distiller's manifest, lock and install config,
    # a package distillation does not import, and caches, dotfiles, editor leftovers and prose.
    before = code_vintage()
    _write(
        package,
        {
            "__pycache__/hydrate.pyc": "bytes",
            ".DS_Store": "finder",
            "hydrate.py~": "an editor's backup",
            "#hydrate.py#": "an editor's autosave",
            "README.md": "prose",
        },
    )
    _write(
        distill,
        {
            "package.json": DISTILL["package.json"].replace("2.1.7", "2.1.8"),
            "bun.lock": "another lock\n",
            "bunfig.toml": "[install]\nminimumReleaseAge = 1\n",
            "node_modules/rrweb/package.json": DISTILL[
                "node_modules/rrweb/package.json"
            ].replace("2.1.7", "2.1.8"),
        },
    )
    assert code_vintage() == before


def test_what_string_raw_reads_counts(code):
    _, distill = code
    _write(distill, {"distill.js": "export const logic = String.raw`a\\nb`;\n"})
    escaped = code_vintage()
    _write(distill, {"distill.js": "export const logic = String.raw`a\nb`;\n"})
    assert code_vintage() != escaped


def test_every_branch_counts_whatever_the_reading_environment(code, monkeypatch):
    # The transpiler would otherwise write in the reader's NODE_ENV and drop the branch it rules out.
    _, distill = code
    branch = (
        'export const mode = process.env.NODE_ENV === "production" ? "{}" : "dev";\n'
    )
    _write(distill, {"distill.js": branch.format("a")})
    first = code_vintage()
    monkeypatch.setenv("NODE_ENV", "production")
    assert code_vintage() == first
    monkeypatch.delenv("NODE_ENV")
    _write(distill, {"distill.js": branch.format("b")})
    assert code_vintage() != first


def test_a_package_distillation_imports_counts_by_its_installed_version(code):
    _, distill = code
    first = code_vintage()
    _write(
        distill,
        {
            "node_modules/aria-query/package.json": DISTILL[
                "node_modules/aria-query/package.json"
            ].replace("5.3.2", "5.3.3")
        },
    )
    assert code_vintage() != first


def test_a_package_distillation_imports_not_installed_makes_the_vintage_unreadable(
    code,
):
    _, distill = code
    (distill / "node_modules" / "aria-query" / "package.json").unlink()
    with pytest.raises(
        VintageUnreadable, match="aria-query, which distillation imports"
    ):
        code_vintage()


@pytest.mark.parametrize(
    ("where", "edit", "finding"),
    [
        (
            "distill",
            {"distill.js": "export const logic = ;\n"},
            r"JavaScript failed: .*distill\.js:1:\d+: ",
        ),
        ("package", {"hydrate.py": "def load(event)\n"}, "does not parse.*hydrate.py"),
    ],
)
def test_code_mid_edit_makes_the_vintage_unreadable(code, where, edit, finding):
    package, distill = code
    _write({"package": package, "distill": distill}[where], edit)
    with pytest.raises(VintageUnreadable, match=finding):
        code_vintage()


def test_the_real_code_vintage_covers_the_package_and_the_distiller():
    files = derive_mod._code_files()
    assert {
        "evidence/hydrate.py",
        "evidence/db.py",
        "evidence/replay_seek.js",
        "distill/distill.js",
        "distill/rescue_gate.js",
    } <= files.keys()
    assert not any("node_modules" in key or "__pycache__" in key for key in files)
    assert {path.suffix for path in files.values()} == {".py", ".js"}


def _stamped_db(tmp_path):
    db = str(tmp_path / "events.db")
    conn = connect(db)
    hydrate(conn, *read_recording(FIXTURE))
    derive_mod.finish_derivations(conn, db, DEFINITION)
    return conn, DEFINITION


def test_a_definition_change_heals_sessions_alone_and_a_code_change_everything(
    tmp_path, monkeypatch
):
    conn, definition = _stamped_db(tmp_path)
    assert derive_mod.stale_stage(conn, definition) is None

    adjusted = {
        **definition,
        "session": {**definition["session"], "inactivity_minutes": 45},
    }
    assert derive_mod.stale_stage(conn, adjusted) == "sessions"
    assert derive_mod.stale_stage(conn, {**definition, "path": "elsewhere"}) is None

    monkeypatch.setattr(derive_mod, "_LOADED_CODE", "other code")
    assert derive_mod.stale_stage(conn, definition) == "timestamps"
    assert derive_mod.stale_stage(conn, adjusted) == "timestamps"


def test_the_code_identity_is_the_one_read_as_the_process_loaded_the_code(
    code, tmp_path, monkeypatch
):
    # Read once at load: an edit after it moves nothing this process reads or stamps, and the
    # next process, reading the edited code, finds the db stale.
    package, _ = code
    monkeypatch.setattr(derive_mod, "_LOADED_CODE", derive_mod._read_code_vintage())
    loaded = derive_mod._LOADED_CODE
    _write(package, {"hydrate.py": PACKAGE["hydrate.py"] + "\nLATER = 1\n"})
    assert derive_mod.installed_vintages(DEFINITION)["code"] == loaded

    db = str(tmp_path / "events.db")
    conn = connect(db)
    hydrate(conn, *read_recording(FIXTURE))
    derive_mod.finish_derivations(conn, db, DEFINITION)
    assert derive_mod.stamped_vintages(conn)["code"] == loaded

    monkeypatch.setattr(derive_mod, "_LOADED_CODE", derive_mod._read_code_vintage())
    assert derive_mod.stale_stage(conn, DEFINITION) == "timestamps"


def test_code_unreadable_at_load_exits_with_the_reason(monkeypatch):
    monkeypatch.setattr(
        derive_mod,
        "_LOADED_CODE",
        VintageUnreadable("reading the derivation's JavaScript failed"),
    )
    with pytest.raises(SystemExit, match="JavaScript failed"):
        derive_mod.installed_vintages(DEFINITION)
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    with pytest.raises(SystemExit, match="no bun on PATH"):
        derive_mod.installed_vintages(DEFINITION)


@pytest.mark.parametrize(
    "edit",
    [
        {"node_modules/aria-query/package.json": "{"},
        {"node_modules/aria-query/package.json": '{"name": "aria-query"}\n'},
    ],
)
def test_any_failure_to_read_the_code_makes_the_vintage_unreadable(code, edit):
    _, distill = code
    _write(distill, edit)
    with pytest.raises(
        VintageUnreadable, match="aria-query/package.json states no version"
    ):
        code_vintage()


def test_the_stamp_holds_exactly_the_installed_parts(tmp_path):
    # Read and written as one set of rows, so a part stamped but no longer installed reads as
    # stale, and the heal leaves the stamp equal to what is installed.
    conn, definition = _stamped_db(tmp_path)
    conn.execute("INSERT INTO meta (key, value) VALUES ('vintage_retired', 'x')")
    conn.commit()
    assert derive_mod.stale_stage(conn, definition) == "timestamps"
    derive_mod.finish_derivations(conn, str(tmp_path / "events.db"), definition)
    assert derive_mod.stamped_vintages(conn) == derive_mod.installed_vintages(
        definition
    )
    assert derive_mod.stale_stage(conn, definition) is None


def test_a_definition_change_still_brings_new_rows_through_every_pass(tmp_path):
    conn, definition = _stamped_db(tmp_path)
    _, events = read_recording(FIXTURE)
    hydrate(conn, "a-new-visitor", events)
    adjusted = {
        **definition,
        "session": {**definition["session"], "inactivity_minutes": 45},
    }
    done = derive_mod.finish_derivations(conn, str(tmp_path / "events.db"), adjusted)
    assert done["from"] == "sessions" and "a-new-visitor" in done["dirty"]
    assert derive_mod.stranded_events(conn) == 0
    assert derive_mod.stale_stage(conn, adjusted) is None


def test_a_definition_change_re_derives_sessions_and_nothing_before_them(tmp_path):
    conn, definition = _stamped_db(tmp_path)
    # A timestamp and a distilled column set wrong by hand: a pass that re-derived either stage
    # would put them right.
    event = conn.execute(
        "SELECT id, timestamp FROM events ORDER BY timestamp DESC, id DESC LIMIT 1"
    ).fetchone()
    conn.execute(
        "UPDATE events SET timestamp = timestamp + 12345 WHERE id = ?", (event["id"],)
    )
    distilled = conn.execute(
        "SELECT id FROM events WHERE type_str IS NOT NULL ORDER BY id LIMIT 1"
    ).fetchone()
    conn.execute(
        "UPDATE events SET type_str = 'set by hand' WHERE id = ?", (distilled["id"],)
    )
    conn.commit()
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    adjusted = {
        **definition,
        "session": {**definition["session"], "inactivity_minutes": 0},
    }
    derive_mod.finish_derivations(conn, str(tmp_path / "events.db"), adjusted)
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] > sessions, (
        "a zero gap makes every act its own session"
    )
    assert (
        conn.execute(
            "SELECT timestamp FROM events WHERE id = ?", (event["id"],)
        ).fetchone()[0]
        == event["timestamp"] + 12345
    )
    assert (
        conn.execute(
            "SELECT type_str FROM events WHERE id = ?", (distilled["id"],)
        ).fetchone()[0]
        == "set by hand"
    )
    assert derive_mod.stale_stage(conn, adjusted) is None
