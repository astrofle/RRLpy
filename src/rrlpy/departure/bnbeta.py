"""
Departure coefficient class.
"""

import warnings

import numpy as np
from scipy.interpolate import RectBivariateSpline


class BnBeta:
    """ """

    def __init__(self, n, bn, te, ne, tr, beta=None, transition=None, element=None, frequency=None):
        self.n = n
        self.bn = bn
        self.ne = ne
        self.te = te
        self.tr = tr
        self.beta = beta
        self.transition = transition
        self.element = element
        self.frequency = frequency

        self.indices = None
        self.mask = None

    def set_indices(self, n):
        """
        Finds the indices for the departure coefficients given
        the principal quantum numbers `n`.

        Parameters
        ----------
        n : list
            Principal quantum numbers.

        Returns
        -------

        """

        self.indices = np.array([np.argmin(abs(n_ - self.n)) for n_ in n])

    def select(self, ne, te, tr=None):
        """ """

        if tr is not None:
            mask = (self.ne == ne) & (self.te == te) & (self.tr == tr)
        else:
            mask = (self.ne == ne) & (self.te == te)

        if mask.sum() == 0:
            raise ValueError("No departure coefficients for the specified physical conditions")
        self.mask = mask

    def get_bn(self, ne, te, tr):
        """ """

        self.select(ne, te, tr)

        return self.bn[self.mask][0, self.indices]

    def get_bm(self, ne, te, tr):
        """ """

        self.select(ne, te, tr)

        dn = 1

        return self.bn[self.mask][0, self.indices + dn]

    def get_beta(self, ne, te, tr):
        """ """

        self.select(ne, te, tr)

        return self.beta[self.mask][0, self.indices]

    def interpolate(self, n=None):
        """ """

        if n is None:
            if self.indices is not None:
                idx = self.indices
            else:
                warnings.warn("Will use all n values.")
                idx = np.arange(len(self.n) - 1)
        else:
            self.set_indices(n)
            idx = self.indices

        bn_mod_fun = {}
        beta_mod_fun = {}

        te_grid = np.unique(self.te)
        ne_grid = np.unique(self.ne)

        for i in idx:
            n = self.n[i]
            bn_mod_fun[n] = RectBivariateSpline(te_grid, ne_grid, self.bn.T[i, :].reshape((len(te_grid), len(ne_grid))))
            beta_mod_fun[n] = RectBivariateSpline(
                te_grid, ne_grid, self.beta.T[i, :].reshape((len(te_grid), len(ne_grid)))
            )

        return BnBetaInterp(
            self.n[idx],
            bn_mod_fun,
            te_grid,
            ne_grid,
            None,
            beta=beta_mod_fun,
            transition=self.transition,
            element=self.element,
            frequency=self.frequency,
        )


class BnBetaInterp:
    def __init__(self, n, bn, te, ne, tr, beta=None, transition=None, element=None, frequency=None):
        self.n = n
        self.bn = bn
        self.ne = ne
        self.te = te
        self.tr = tr
        self.beta = beta
        self.transition = transition
        self.element = element
        self.frequency = frequency

        self.indices = None
        self.mask = None
        self._n = n

    def set_indices(self, n):
        self._n = n

    def get_bn(self, ne, te, tr=None):
        bn_out = np.zeros(len(self._n), dtype="d")
        for i, n in enumerate(self._n):
            bn_out[i] = self.bn[n](te, ne)

        return bn_out

    def get_bm(self, ne, te, tr=None):
        bm_out = np.zeros(len(self._n), dtype="d")
        for i, n in enumerate(self._n):
            bm_out[i] = self.bn[n + 1](te, ne)

        return bm_out

    def get_beta(self, ne, te, tr=None):
        beta_out = np.zeros(len(self._n), dtype="d")
        for i, n in enumerate(self._n):
            beta_out[i] = self.beta[n](te, ne)

        return beta_out
