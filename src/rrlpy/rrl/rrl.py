"""
RRL class.
"""

import numpy as np

from rrlpy import freq

from . import constants, core


class RRLs:
    """ """

    def __init__(self, qns, elements, transitions, zs, amplitudes=None, widths=None, centers=None, frequencies=None):
        self.qn = qns
        self.element = elements
        self.transition = transitions
        self.z = zs
        self.amplitude = amplitudes
        self.width = widths
        self.center = centers
        self.frequency = frequencies

        self.dn = np.array([freq.transition2dn(t) for t in self.transition])

        if self.frequency is None:
            self._qn2freq()

        self.fnnp = np.array([core.fnnp_app(qn, dn) for qn, dn in zip(self.qn, self.dn)])
        self.mass = np.array([constants.mass[e] for e in self.element])

    def _qn2freq(self):
        self.frequency = np.array(
            [
                freq.frequency(qn + dn, qn, z=z, m=constants.mass[e])
                for qn, dn, z, e in zip(self.qn, self.dn, self.z, self.element)
            ]
        )
