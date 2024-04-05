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
