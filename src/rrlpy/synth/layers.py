"""
Layer class.
"""

import numpy as np
from astropy import constants as ac
from astropy import units as u

from rrlpy import continuum, rrls


class Layers:
    def __init__(self, ne, te, l, v_rms, departure_coefs, rrls, background, medium):
        self.ne_vals = ne
        self.te_vals = te
        self.ds_vals = l
        self.vrms_vals = v_rms
        self.bnbeta = departure_coefs
        self.rrls = rrls
        self.background = background
        self.medium = medium
        self._freq = self.rrls.get_freq()

        # Infer.
        self.n_layers = len(ne)
        self.tau_c = np.zeros((self.nlayers, len(rrls.qns)), dtype="d")
        self.tau_l = np.zeros((self.nlayers, len(rrls.qns)), dtype="d")
        self.t_sum = np.zeros((self.nlayers, len(rrls.qns)), dtype="d") * u.K
        self.t_cont = np.zeros((self.nlayers, len(rrls.qns)), dtype="d") * u.K
        self.t_line = np.zeros((self.nlayers, len(rrls.qns)), dtype="d") * u.K
        self.t_tot_out = np.zeros((len(rrls.qns)), dtype="d") * u.K
        self.t_cont_out = np.zeros((len(rrls.qns)), dtype="d") * u.K

    def compute(self):
        """ """

        for i in range(self.n_layers):
            ne = self.ne_vals[i]
            te = self.te_vals[i]
            ds = self.ds_vals[i]
            dv = self.vrms_vals[i]

            bnl = self.bnbeta.get_bn(ne, te, self.background.get_tr100())
            bml = self.bnbeta.get_bm(ne, te, self.background.get_tr100())
            betal = self.bnbeta.get_beta(ne, te, self.background.get_tr100())

            # Compute continuum optical depth for the layer.
            self.tau_c[i] = continuum.tau(self._freq, te, ne, ne, ds, z=self.rrls.get_z()).cgs

            # Compute line width for layer.
            dnu = dv / ac.c * self._freq
            phi = 1.0 / (1.064 * dnu)
            # RRL optical depth.
            self.tau_l[i] = (
                rrls.tau_exact(
                    self.rrls.qns,
                    ne,
                    te,
                    ne,
                    ds,
                    self.rrls.fnnp,
                    self._freq,
                    dn=self.rrls.dn,
                    z=self.rrls.get_z(),
                )
                * phi
            ).cgs

            # First layer.
            if i == 0:
                tmi = self.medium.eval(self._freq)
                t0ci = self.background.eval(self._freq)
                t0li = self.background.eval(self._freq)
            else:
                tmi = self.medium.eval(self._freq)
                t0ci = self.t_cont[i - 1]
                t0li = self.t_tot[i - 1]

            # Line emission.
            self.t_tot[i] = rrls.layer_emission(
                self.tau_l[i], self.tau_c[i], bnl, bml, betal, te, t0=t0li, tm=tmi, tf=0 * u.K
            )
            # Continuum emission.
            self.t_cont[i] = continuum.continuum_brightness(te, t0ci, tmi, 0 * u.K, self.tau_c[i])
            self.t_line[i] = self.t_tot[i] - self.t_cont[i]
