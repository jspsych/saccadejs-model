"""Regression tests for Keras backend selection at package-import time.

Background: ``et_util/__init__.py`` imports ``et_util.custom_layers``, whose first
line is ``import keras``. Python runs the package __init__ before the body of any
submodule, so under ``python -m et_util.train`` the keras backend is already locked
in by the time train.py's own ``os.environ.setdefault("KERAS_BACKEND", "jax")``
runs -- making it dead code. The observed consequence was training silently running
on tensorflow-cpu at 109 s/step instead of jax-TPU at 0.3 s/step (~360x).

The fix puts the setdefault at the very top of ``et_util/__init__.py``, before every
import. These tests guard that ordering.

Run with::

    pytest tests/test_keras_backend.py

or standalone (no pytest required)::

    python tests/test_keras_backend.py

Each backend check runs in a *fresh subprocess* with ``KERAS_BACKEND`` scrubbed from
the environment, so the test runner's own backend cannot pollute the result.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INIT_PATH = REPO_ROOT / "et_util" / "__init__.py"

# Printed by the child when keras cannot initialize (e.g. jax not installed here).
_UNAVAILABLE = "BACKEND_UNAVAILABLE"

_PROBE = f"""
try:
    import et_util  # noqa: F401
    import keras
    print("BACKEND=" + keras.backend.backend())
except Exception as exc:
    print("BACKEND={_UNAVAILABLE}:" + type(exc).__name__ + ": " + str(exc)[:200])
"""


def _probe_backend(backend_env=None):
    """Import et_util in a clean subprocess and report the resulting keras backend.

    ``backend_env=None`` scrubs KERAS_BACKEND entirely; otherwise it is exported.
    """
    env = dict(os.environ)
    env.pop("KERAS_BACKEND", None)
    if backend_env is not None:
        env["KERAS_BACKEND"] = backend_env
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("BACKEND="):
            return line[len("BACKEND=") :]
    raise AssertionError(
        f"probe produced no BACKEND line\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def _skip_or_fail(result, expected):
    """Skip when the backend simply is not installed; otherwise it is a real failure."""
    if result.startswith(_UNAVAILABLE):
        msg = f"{expected!r} backend not importable in this env: {result}"
        try:
            import pytest

            pytest.skip(msg)
        except ImportError:
            print(f"SKIP: {msg}")
        return True
    return False


def test_setdefault_precedes_all_imports_in_package_init():
    """Dependency-free guard: the setdefault must come before any import in __init__.

    This is the assertion that actually catches a reintroduction of the bug, and it
    runs even in environments where neither jax nor tensorflow is installed.
    """
    lines = INIT_PATH.read_text().splitlines()

    setdefault_line = None
    first_import_line = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        if setdefault_line is None and "KERAS_BACKEND" in line and "setdefault" in line:
            setdefault_line = i
        if (
            first_import_line is None
            and (line.startswith("import ") or line.startswith("from "))
            # `import os` is required to call setdefault at all, so it is exempt.
            and line != "import os"
        ):
            first_import_line = i

    assert setdefault_line is not None, (
        f"{INIT_PATH} no longer sets KERAS_BACKEND; every submodule that imports "
        "keras would then get keras's default backend"
    )
    assert first_import_line is not None, "expected et_util/__init__.py to import things"
    assert setdefault_line < first_import_line, (
        "os.environ.setdefault('KERAS_BACKEND', ...) must precede every import in "
        f"{INIT_PATH} (found at line {setdefault_line + 1}, but an import appears at "
        f"line {first_import_line + 1}). et_util.custom_layers imports keras, which "
        "locks the backend in at import time, so a later setdefault is dead code."
    )


def test_default_backend_is_jax_when_env_unset():
    result = _probe_backend(backend_env=None)
    if _skip_or_fail(result, "jax"):
        return
    assert result == "jax", (
        f"importing et_util with KERAS_BACKEND unset selected {result!r}, not 'jax'. "
        "This is the ~360x slowdown bug: keras fell back to tensorflow."
    )


def test_explicit_env_var_overrides_the_default():
    result = _probe_backend(backend_env="tensorflow")
    if _skip_or_fail(result, "tensorflow"):
        return
    assert result == "tensorflow", (
        f"KERAS_BACKEND=tensorflow was overridden to {result!r}; the eval-only "
        "analysis scripts depend on an explicit env var winning over the setdefault."
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name}")
    sys.exit(1 if failures else 0)
