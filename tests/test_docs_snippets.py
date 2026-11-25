"""
Execute the Python code blocks in the Sphinx docs.

Each ``.rst`` page runs top to bottom in one namespace, as a reader would
paste it, in a temporary working directory. Blocks whose directive carries
``:class: no-exec`` are skipped. matplotlib is headless and Plotly figures
are never shown, so no windows or browser tabs open.

Run only these tests with ``pytest -m docs``; deselect them with
``-m "not docs"``.
"""

import os
import re
import warnings
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
DIRECTIVE = re.compile(r"^(\s*)\.\. code-block:: python\s*$")
OPTION = re.compile(r"^\s+:[\w-]+:")

PAGES = sorted(
    p for p in DOCS.rglob("*.rst")
    if "_build" not in p.parts
    and any(DIRECTIVE.match(line) for line in p.read_text().splitlines())
)


def code_blocks(text):
    """Yield (line number, source, skip) for each python code block."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not DIRECTIVE.match(lines[i]):
            i += 1
            continue
        j = i + 1
        skip = False
        while j < len(lines) and OPTION.match(lines[j]):
            skip = skip or "no-exec" in lines[j]
            j += 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            return
        indent = len(lines[j]) - len(lines[j].lstrip())
        body = []
        k = j
        while k < len(lines) and (not lines[k].strip() or
                                  len(lines[k]) - len(lines[k].lstrip()) >= indent):
            body.append(lines[k][indent:] if lines[k].strip() else "")
            k += 1
        yield j + 1, "\n".join(body), skip
        i = k


@pytest.fixture
def headless(monkeypatch, tmp_path):
    """Headless plotting in a scratch directory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setenv("PLOTLY_RENDERER", "json")
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        monkeypatch.setattr(pio.renderers, "default", "json")
        monkeypatch.setattr(pio, "show", lambda *a, **k: None)
        monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    except ImportError:
        pass
    monkeypatch.chdir(tmp_path)
    yield
    plt.close("all")


@pytest.mark.docs
@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_docs_page_runs(page, headless):
    import matplotlib.pyplot as plt

    namespace = {"__name__": "__docs__"}
    failures = []
    for lineno, source, skip in code_blocks(page.read_text()):
        if skip:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(compile(source, f"{page.name}:{lineno}", "exec"), namespace)
        except Exception as exc:  # collect every failing block on the page
            failures.append(f"line {lineno}: {type(exc).__name__}: {exc}")
        finally:
            plt.close("all")
    assert not failures, f"{page.relative_to(DOCS)}:\n" + "\n".join(failures)


def test_pages_found():
    assert len(PAGES) >= 10, os.fspath(DOCS)
