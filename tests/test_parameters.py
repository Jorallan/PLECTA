"""The parameter file is the configuration; these guard that claim."""
from dataclasses import fields

import pytest

from plecta.depth import DepthParams
from plecta.image.bundles import SampleParams
from plecta.image.measurement import CutParams
from plecta.image.refine import RefineParams
from plecta.joint import JointParams
from plecta.linking import Params
from plecta.parameters import SECTION_FOR, build, flatten, verify_against_frozen

ALL = (Params, DepthParams, JointParams, SampleParams, CutParams, RefineParams)


def test_grouping_matches_the_frozen_published_set():
    """The one check that protects the manuscript's numbers."""
    verify_against_frozen()


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
def test_every_field_is_in_the_file(cls):
    """No parameter may be tunable in code but absent from parameters.yaml.

    Otherwise reading the file would not tell you the configuration, which is
    the whole point of having it.
    """
    assert {f.name for f in fields(cls)} == set(flatten(SECTION_FOR[cls.__name__]))


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
def test_builds_and_preserves_types(cls):
    built, default = build(cls), cls()
    for f in fields(cls):
        assert type(getattr(built, f.name)) is type(getattr(default, f.name)), f.name


def test_override_beats_the_file():
    assert build(Params, ["spur_px=9"]).spur_px == 9


def test_unknown_key_is_refused():
    with pytest.raises(SystemExit):
        build(Params, ["not_a_parameter=1"])
