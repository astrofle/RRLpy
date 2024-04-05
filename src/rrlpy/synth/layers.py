"""
Layer class.
"""

import numpy as np
from astropy import constants as ac
from astropy import units as u

from rrlpy import continuum, linewidth, rrl


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
        self._freq = self.rrls.frequency * u.Hz

        # Infer.
        self.n_layers = len(ne)
        self.dv = np.zeros((self.n_layers, len(rrls.qn)), dtype="d")
        self.tau_c = np.zeros((self.n_layers, len(rrls.qn)), dtype="d")
        self.tau_l = np.zeros((self.n_layers, len(rrls.qn)), dtype="d")
        self.t_tot = np.zeros((self.n_layers, len(rrls.qn)), dtype="d") * u.K
        self.t_cont = np.zeros((self.n_layers, len(rrls.qn)), dtype="d") * u.K
        self.t_line = np.zeros((self.n_layers, len(rrls.qn)), dtype="d") * u.K
        self.t_tot_out = np.zeros((len(rrls.qn)), dtype="d") * u.K
        self.t_cont_out = np.zeros((len(rrls.qn)), dtype="d") * u.K

        self.use_tr_bn = True
        if self.bnbeta.tr is None:
            self.use_tr_bn = False

        self.bnbeta.set_indices(self.rrls.qn)

    def compute(self):
        """ """

        tr100 = None
        if self.use_tr_bn:
            tr100 = self.background.get_tr100()

        for i in range(self.n_layers):
            ne = self.ne_vals[i]
            te = self.te_vals[i]
            ds = self.ds_vals[i]
            dv = self.vrms_vals[i]

            bnl = self.bnbeta.get_bn(ne, te, tr100)
            bml = self.bnbeta.get_bm(ne, te, tr100)
            betal = self.bnbeta.get_beta(ne, te, tr100)

            # Compute continuum optical depth for the layer.
            self.tau_c[i] = continuum.tau(self._freq, te, ne, ne, ds, z=self.rrls.z).cgs

            # Compute line width for layer.
            dv_gauss = (
                linewidth.doppler_broad(te.to("K").value, self.rrls.mass, dv.to("m/s").value, fwhm=True) * u.m / u.s
            )
            dv_press = (
                linewidth.pressure_broad_salgado(self.rrls.qn, te.to("K").value, ne.to("cm-3").value, self.rrls.dn)
                * u.Hz
            )
            dv_press_v = dv_press / self._freq * ac.c
            dv_radi = (
                linewidth.radiation_broad_salgado(self.rrls.qn, 1.0, self.background.get_tr100().to("K").value) * u.Hz
            )
            dv_radi_v = dv_radi / self._freq * ac.c
            dv_lrntz = dv_press_v + dv_radi_v
            self.dv = linewidth.voigt_fwhm(dv_gauss, dv_lrntz)
            # Convert the linewidth to frequency.
            dnu = self.dv / ac.c * self._freq
            # Estimate the peak of the line.
            phi = 1.0 / (1.064 * dnu)
            # RRL optical depth.
            self.tau_l[i] = (
                rrl.tau_exact(
                    self.rrls.qn,
                    ne,
                    te,
                    ne,
                    ds,
                    self.rrls.fnnp,
                    self._freq,
                    dn=self.rrls.dn,
                    z=self.rrls.z,
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
            self.t_tot[i] = self.layer_emission(
                self.tau_l[i], self.tau_c[i], bnl, bml, betal, te, t0=t0li, tm=tmi, tf=0 * u.K
            )
            # Continuum emission.
            self.t_cont[i] = continuum.continuum_brightness(te, t0ci, tmi, 0 * u.K, self.tau_c[i])
            self.t_line[i] = self.t_tot[i] - self.t_cont[i]

    def layer_emission(self, tau_l, tau_c, bn, bm, beta, te, t0=0, tm=0, tf=0):
        """ """

        tau_l_nolte = bn * beta * tau_l
        tau_tot = tau_l_nolte + tau_c

        return t0 * np.exp(-tau_tot) + (te * (bm * tau_l + tau_c) + tm) / (tau_tot) * (1.0 - np.exp(-tau_tot))
