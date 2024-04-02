"""
Departure coefficient class.
"""

import numpy as np


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
