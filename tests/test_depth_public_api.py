"""Behavioral contracts for the public depth helpers.

Checks layer assignment and cycle grouping for downstream callers.

    python -m unittest tests.test_depth_public_api -v
"""
from __future__ import annotations

import unittest

from plecta import depth as D


def _decided(i: int, j: int, over: int, *, raw_over: int | None = None):
    """A crossing whose local and global calls are both `over`, unless split."""
    c = D.PredCrossing(i=i, j=j, x=0.0, y=0.0)
    c.raw_over = over if raw_over is None else raw_over
    c.over = over
    return c


def _abstained(i: int, j: int):
    c = D.PredCrossing(i=i, j=j, x=0.0, y=0.0)
    c.abstain = True
    return c


class TestStackLevels(unittest.TestCase):
    """The layer count over EVERY constraint, not only the decided ones."""

    def test_decided_chain_gives_one_level_per_rod(self):
        ids = [1, 2, 3]
        crossings = [_decided(1, 2, 1), _decided(2, 3, 2)]
        levels, n = D.stack_levels(ids, crossings)
        self.assertEqual(n, 3)
        self.assertLess(levels[3], levels[2])
        self.assertLess(levels[2], levels[1])

    def test_abstained_pair_still_separates_via_the_global_order(self):
        """This is the whole reason the function is not `assign_layers`."""
        ids = [1, 2]
        crossings = [_abstained(1, 2)]

        levels, n = D.stack_levels(ids, crossings)
        self.assertEqual(n, 1, "no evidence and no order: one flat layer")

        levels, n = D.stack_levels(ids, crossings, order_position={1: 0, 2: 1})
        self.assertEqual(n, 2, "the order supplies the direction")
        self.assertLess(levels[2], levels[1])

    def test_never_reports_fewer_levels_than_assign_layers(self):
        ids = [1, 2, 3]
        crossings = [_decided(1, 2, 1), _abstained(2, 3)]
        _, k = D.assign_layers(ids, crossings)
        _, n = D.stack_levels(ids, crossings, order_position={1: 0, 2: 1, 3: 2})
        self.assertGreaterEqual(n, k)


class TestRawCycleGroups(unittest.TestCase):
    """Which instances are in a contradictory group, on the RAW calls."""

    def test_consistent_evidence_has_no_group(self):
        ids = [1, 2, 3]
        crossings = [_decided(1, 2, 1), _decided(2, 3, 2)]
        self.assertEqual(D.raw_cycle_groups(ids, crossings), {})

    def test_a_three_cycle_is_reported_as_one_group(self):
        ids = [1, 2, 3]
        crossings = [_decided(1, 2, 1), _decided(2, 3, 2), _decided(3, 1, 3)]
        groups = D.raw_cycle_groups(ids, crossings)
        self.assertEqual(set(groups), {1, 2, 3})
        self.assertEqual(len(set(groups.values())), 1)

    def test_it_reads_the_raw_call_not_the_corrected_one(self):
        """The groups describe the evidence, not the fix applied to it."""
        ids = [1, 2, 3]
        crossings = [_decided(1, 2, 1), _decided(2, 3, 2),
                     # the solver reversed this one to break the cycle, but the
                     # raw local call still points 3 over 1
                     _decided(3, 1, 1, raw_over=3)]
        self.assertEqual(set(D.raw_cycle_groups(ids, crossings)), {1, 2, 3})

    def test_abstained_crossings_carry_no_constraint(self):
        ids = [1, 2, 3]
        crossings = [_decided(1, 2, 1), _decided(2, 3, 2), _abstained(3, 1)]
        self.assertEqual(D.raw_cycle_groups(ids, crossings), {})


if __name__ == "__main__":
    unittest.main()
