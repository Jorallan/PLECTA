"""Read every tunable parameter from `parameters.yaml`.

One file holds the whole adjustable surface of the package. This module turns a
section of it into the dataclass that stage expects, and nothing else in the
package should read a tuned value from anywhere.

The dataclass defaults are deliberately left in place: they keep each class
constructible on its own, which the tests rely on. They are not the tuned
values and must not be treated as such -- `linking.Params.join_px` defaults to
0 and is tuned to 14. Anything that builds a params object for real work goes
through here.

`grouping_2d` is additionally checked against `params.json`, the frozen record
of the configuration that produced every published number. The check is not
decoration: the YAML is editable by design, and a stray edit to that section
would silently invalidate the manuscript's numbers. `verify_against_frozen()`
fails loudly instead, and the test suite calls it.
"""
from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar

CONFIG_PATH = Path(__file__).resolve().parent / "parameters.yaml"
FROZEN_PATH = Path(__file__).resolve().parent / "params.json"

T = TypeVar("T")

# Which YAML section feeds which dataclass. Keyed by class name so a caller
# never has to remember the section string.
SECTION_FOR = {
    "Params": "grouping_2d",
    "DepthParams": "depth_3d",
    "JointParams": "joint_greyscale",
    "SampleParams": "width_sampling",
    "CutParams": "width_profile",
    "RefineParams": "rendering",
}


@lru_cache(maxsize=1)
def _raw() -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "PLECTA reads its parameters from parameters.yaml and needs "
            "PyYAML. Install it with `pip install pyyaml`."
        ) from exc
    if not CONFIG_PATH.is_file():
        raise RuntimeError(f"PLECTA parameter file is missing: {CONFIG_PATH}")
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{CONFIG_PATH} did not parse to a mapping")
    return data


def flatten(section: str) -> Dict[str, Any]:
    """One section's parameters, with its grouping headings removed.

    The groups exist to make the file readable; the dataclasses are flat, so a
    duplicated name across two groups would resolve arbitrarily. That is
    refused rather than resolved.
    """
    raw = _raw()
    if section not in raw:
        raise RuntimeError(
            f"{CONFIG_PATH} has no section '{section}'. "
            f"present: {sorted(raw)}")
    out: Dict[str, Any] = {}
    for group, entries in (raw[section] or {}).items():
        if not isinstance(entries, dict):
            raise RuntimeError(
                f"{section}.{group} should be a mapping of parameters")
        for key, value in entries.items():
            if key in out:
                raise RuntimeError(
                    f"{section}: '{key}' appears in more than one group")
            out[key] = value
    return out


def build(cls: Type[T], overrides: List[str] | None = None,
          section: str | None = None) -> T:
    """Construct `cls` from its YAML section, then apply CLI overrides.

    Types are taken from the dataclass *defaults*, not the annotations: these
    modules use `from __future__ import annotations`, so a field's .type is the
    string "int" and an integer parameter would silently become a float.
    """
    section = section or SECTION_FOR.get(cls.__name__)
    if section is None:
        raise RuntimeError(f"no parameters.yaml section is mapped to {cls.__name__}")

    defaults = cls()
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    values = dict(flatten(section))

    unknown = set(values) - known
    if unknown:
        raise RuntimeError(
            f"{CONFIG_PATH} section '{section}' sets {sorted(unknown)}, "
            f"which {cls.__name__} does not define")
    absent = known - set(values)
    if absent:
        raise RuntimeError(
            f"{CONFIG_PATH} section '{section}' is missing {sorted(absent)}. "
            "Every tunable parameter must appear in the file, so that reading "
            "the file tells you the whole configuration.")

    for item in overrides or []:
        key, _, raw = item.partition("=")
        key = key.strip()
        if key not in known:
            raise SystemExit(
                f"unknown parameter '{key}'. known: {sorted(known)}")
        values[key] = raw

    coerced = {}
    for key, value in values.items():
        proto = getattr(defaults, key)
        if isinstance(proto, bool):
            coerced[key] = (value.strip().lower() in ("1", "true", "yes")
                            if isinstance(value, str) else bool(value))
        elif isinstance(proto, str):
            coerced[key] = str(value)
        elif isinstance(proto, int):
            coerced[key] = int(value)
        else:
            coerced[key] = float(value)
    return cls(**coerced)


def verify_against_frozen() -> None:
    """Assert the grouping section still equals the frozen published set.

    Raises with the offending keys rather than returning a bool, because the
    only useful response to a mismatch is to look at what moved.
    """
    import json

    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    live = flatten("grouping_2d")
    drift = {k: (frozen[k], live.get(k)) for k in frozen
             if k not in live or float(live[k]) != float(frozen[k])}
    extra = sorted(set(live) - set(frozen))
    if drift or extra:
        raise AssertionError(
            "parameters.yaml `grouping_2d` no longer matches the frozen "
            f"params.json that produced the published numbers.\n"
            f"  changed (frozen -> yaml): {drift}\n"
            f"  present only in yaml: {extra}")
