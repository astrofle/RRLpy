import numpy as np

from rrlpy.departure import bnbeta


class TestBnBeta:
    """ """

    def setup_method(self):
        self.ne = np.array([0.1, 1.0, 5.0, 10.0])
        self.te = np.array([1000, 5000, 8000, 10000])
        self.tr = None
        self.n = np.arange(30, 101, 1)
        self.bn = np.ones((len(self.ne), len(self.n)))
        self.bnbeta = bnbeta.BnBeta(self.n, self.bn, self.te, self.ne, self.tr)

    def test_set_indices(self):
        n = np.array([30, 50])
        self.bnbeta.set_indices(n)
        assert (self.bnbeta.n[self.bnbeta.indices] - n).sum() == 0

    def test_get_bn(self):
        te = 5000.0
        ne = 1.0
        tr = None
        bn = self.bnbeta.get_bn(ne, te, tr)
        assert np.all(bn == 1.0)
        n = [30]
        self.bnbeta.set_indices(n)
        bn = self.bnbeta.get_bn(ne, te, tr)
        assert len(bn) == 1
        assert bn.shape == (1,)

    def test_select(self):
        te = 5000.0
        ne = 1.0
        tr = None
        self.bnbeta.select(ne, te, tr)
        assert self.bnbeta.mask.sum() == 1
        assert self.bnbeta.mask[1]

    def test_interpolate(self):
        ne = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        te = np.array(
            [
                6000,
                6000,
                6000,
                6000,
                6000,
                6000,
                7000,
                7000,
                7000,
                7000,
                7000,
                7000,
                8000,
                8000,
                8000,
                8000,
                8000,
                8000,
                9000,
                9000,
                9000,
                9000,
                9000,
                9000,
            ]
        )
        bn = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]] * 4)
        beta = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]] * 4)
        n = np.array([100, 101])

        bnb = bnbeta.BnBeta(n, bn, te, ne, None, beta=beta)

        bnb_i = bnb.interpolate()

        np.testing.assert_array_equal(bnb_i._n, n)
        np.testing.assert_allclose(bnb_i.get_bn(1.5, 7000.0, None), [0.5, 0.5])
