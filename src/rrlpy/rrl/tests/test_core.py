from rrlpy.rrl import core


class TestRRLCore:
    """ """

    def test_mdn(self):
        assert core.mdn(1) == 0.1908
        assert core.mdn(5.0) == 0.001812

    def test_fnnp_app(self):
        assert core.fnnp_app(1, 1) == 0.477
        assert core.fnnp_app(500, 1) == 95.6862
