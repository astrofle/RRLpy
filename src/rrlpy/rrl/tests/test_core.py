import pytest

from astropy import units as u

from rrlpy.rrl import core


class TestRRLCore:
    """ """

    def test_mdn(self):
        assert core.mdn(1) == 0.1908
        assert core.mdn(5.0) == 0.001812

    def test_fnnp_app(self):
        assert core.fnnp_app(1, 1) == 0.477
        assert core.fnnp_app(500, 1) == 95.6862

    def test_xi(self):
        assert core.xi(1, 1.0 * u.K, 1.0).value == pytest.approx(157887.51240204, 1e-8)
