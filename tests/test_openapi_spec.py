"""Catch API-route drift between Flask and docs/api/openapi.yaml.

Fails the build if a Flask route exists in api/app.py but is not
documented in the OpenAPI spec (or vice versa). Path parameters are
normalised: Flask ``/<name>`` ↔ OpenAPI ``/{name}``.

Allow-list known intentional exclusions (SPA fallback, static assets,
etc.) via ``EXCLUDE_FROM_SPEC``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "api" / "app.py"
SPEC_PATH = ROOT / "docs" / "api" / "openapi.yaml"

# Routes that exist in Flask but intentionally aren't in the spec.
EXCLUDE_FROM_SPEC = {
    "/",
    "/<path:path>",                     # SPA fallback
    "/assets/<path:filename>",          # Static bundle
    "/configs",                         # Dev helper, not user-facing
    "/human-game/",                     # Static HTML root
    "/human-game",                      # Static HTML root
    "/human-game/static/<path:filename>",
}


FLASK_ROUTE_RE = re.compile(
    r"@app\.(?:get|post|put|delete|patch)\([\"']([^\"']+)[\"']\)"
)
SPEC_PATH_RE = re.compile(r"^  (/[^\s:]+):\s*$", re.MULTILINE)


def _flask_to_openapi(path: str) -> str:
    """Normalise Flask path-converter syntax to OpenAPI brace syntax."""
    return re.sub(r"<(?:[^:>]+:)?([^>]+)>", r"{\1}", path)


@pytest.fixture(scope="module")
def flask_routes() -> set[str]:
    if not APP_PATH.exists():
        pytest.skip("api/app.py not present")
    raw = APP_PATH.read_text()
    routes = {
        _flask_to_openapi(m.group(1))
        for m in FLASK_ROUTE_RE.finditer(raw)
        if m.group(1) not in EXCLUDE_FROM_SPEC and _flask_to_openapi(m.group(1)) not in EXCLUDE_FROM_SPEC
    }
    return routes


@pytest.fixture(scope="module")
def spec_paths() -> set[str]:
    if not SPEC_PATH.exists():
        pytest.skip("docs/api/openapi.yaml not present")
    raw = SPEC_PATH.read_text()
    return set(SPEC_PATH_RE.findall(raw))


def test_every_flask_post_route_is_documented(flask_routes: set[str], spec_paths: set[str]) -> None:
    """Every POST/PUT/DELETE route in Flask must appear in the spec.

    This is the safety-critical direction: undocumented write endpoints
    are silent surface area for a 3rd-party client."""
    missing = flask_routes - spec_paths
    # Only enforce on the write surface — GETs are less risky and many
    # are auxiliary debug endpoints we deliberately omit.
    write_only = {r for r in missing if any(verb in _verbs_for_route(r) for verb in ("post", "put", "delete"))}
    assert not write_only, (
        f"{len(write_only)} write-side Flask routes are not in docs/api/openapi.yaml:\n"
        + "\n".join(f"  - {r}" for r in sorted(write_only))
        + "\nEither add them to the spec or list them in EXCLUDE_FROM_SPEC."
    )


def test_every_spec_path_exists_in_flask(flask_routes: set[str], spec_paths: set[str]) -> None:
    """Reverse direction: the spec must not document routes that don't exist."""
    extra = spec_paths - flask_routes
    assert not extra, (
        f"{len(extra)} OpenAPI paths have no Flask route:\n"
        + "\n".join(f"  - {p}" for p in sorted(extra))
    )


def _verbs_for_route(route: str) -> set[str]:
    """Return the set of HTTP verbs declared for a given route in app.py."""
    if not APP_PATH.exists():
        return set()
    raw = APP_PATH.read_text()
    flask_form = re.sub(r"\{([^}]+)\}", r"<\1>", route)
    flask_form_typed = re.sub(r"\{([^}]+)\}", r"<path:\1>", route)
    verbs: set[str] = set()
    for verb in ("get", "post", "put", "delete", "patch"):
        pattern = rf"@app\.{verb}\([\"'](?:{re.escape(flask_form)}|{re.escape(route)}|{re.escape(flask_form_typed)})[\"']\)"
        if re.search(pattern, raw):
            verbs.add(verb)
    return verbs
