"""
Tests for rrl.rrl
"""

import pytest

from rrlpy import rrl


class TestRRL:
    def setup_method(self):
        self.qn = [40]
        self.element = ["C"]
        self.transition = ["alpha"]
        self.z = [1]
        self.crrl = rrl.RRLs(self.qn, self.element, self.transition, self.z)
        self.srrl = rrl.RRLs(self.qn, ["S"], self.transition, self.z)

    def test_freq(self):
        assert self.crrl.frequency == pytest.approx([99072.36e6])
        assert self.srrl.frequency == pytest.approx([99075.19e6])
