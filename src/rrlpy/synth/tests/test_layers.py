"""
Test for Layer and Layers.
"""

import numpy as np
from astropy import units as u

from rrlpy.continuum import PowerLaw
from rrlpy.departure import BnBeta
from rrlpy.rrl import RRLs
from rrlpy.synth import layers


class TestLayer:
    def setup_method(self):
        self.ne = 1 * u.cm**-3
        self.te = 1000 * u.K
        self.l = 1 * u.pc
        self.v_rms = 1 * u.km / u.s

        # Set up departure coefficients.
        n = np.arange(30, 100, 1)
        ne = np.array([1]) * u.cm**-3
        te = np.array([1000]) * u.K
        bn = np.ones((1, len(n)))
        beta = np.zeros((1, len(n)))
        self.bnbeta = BnBeta(n, bn, te, ne, None, beta=beta)

        # Set up background radiation field.
        self.background = PowerLaw(1.0 * u.K, 1 * u.GHz, -2.6)
        self.medium = PowerLaw(0.0 * u.K, 1 * u.GHz, -2.6)

        # Set up lines.
        self.rrls = RRLs(
            np.array([30, 60, 90]),
            ["H", "H", "H"],
            ["alpha"] * 3,
            np.array([1, 1, 1]),
        )

        # x-axis
        x_axis = np.arange(-300, 300, 1) * u.km / u.s

        self.layer = layers.Layer(
            self.ne, self.te, self.l, self.v_rms, self.bnbeta, self.rrls, self.background, self.medium, x_axis=x_axis
        )

    def test_compute(self):
        self.layer.compute()


class TestLayers:
    def setup_method(self):
        self.ne = [1] * u.cm**-3
        self.te = [1000] * u.K
        self.l = [1] * u.pc
        self.v_rms = [1] * u.km / u.s

    def test_layers(self):
        pass
