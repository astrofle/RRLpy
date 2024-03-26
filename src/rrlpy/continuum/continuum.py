"""
"""

import numpy as np
from astropy import units as u


class ContinuumMixin:
    def __init__(self):
        self.test = True


class PowerLaw(ContinuumMixin):
    """
    Power law continuum.

    Parameters
    ----------
    t0 : float
        Continuum brightness at nu0.
    nu0 : float
        Reference frequency.
    alpha : float
        Power law index.
    """

    def __init__(self, t0, nu0, alpha):
        self.t0 = t0
        self.nu0 = nu0
        self.alpha = alpha
        self.tr100 = self.get_tr100()

    def eval(self, x):
        """ """

        return self.t0 * np.power(x / self.nu0, self.alpha)

    def get_tr100(self):
        return self.eval(100 * u.MHz)
