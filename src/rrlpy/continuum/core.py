"""Functions for computing continuum strength."""

import numpy as np

from rrlpy.rrl.constants import k_B, m_e, e, c


def gaunt_ff_draine(nu, te, z):
    """
    Free-free emission Gaunt factor from Draine (2011).
    """

    return 13.91 * np.power(z * nu.to("Hz").value, -0.118) * np.power(te.to("K").value, 0.177)


def gaunt_ff_oster(nu, te, z):
    """ """

    return np.log(4.955e-2 * 1.0 / nu.to("GHz").value) + 1.5 * np.log(te.to("K").value)


def continuum_brightness(te, tbg0, tm, tf, tau_c):
    """
    Continuum brightness temperature following the model of Shaver (1975).
    """

    return (
        tbg0 * np.exp(-tau_c)
        + te * (1.0 - np.exp(-tau_c))
        + tm / tau_c * (1.0 - np.exp(-tau_c))
        + tf
    )


def kappa(nu, te, ne, ni, z=1.0, gaunt_fun=gaunt_ff_oster):
    """ """

    gff = gaunt_fun(nu, te, z)

    cte = 8.0 * (e.esu) ** 6.0 / (3.0 * c * np.sqrt(2.0 * np.pi) * np.power(m_e * k_B, 3.0 / 2.0))

    kc = cte * z**2.0 * ne * ni / nu**2.0 * np.power(te, -1.5) * gff

    return kc


def tau(nu, te, ne, ni, pl, z=1.0, gaunt_fun=gaunt_ff_oster):
    """
    Free-free continuum optical depth.

    Parameters
    ----------
    nu : float
        Frequency.
    te : float
        Electron temperature.
    ne : float
        Electron density.
    ni : float
        Collisional partner density.
    z : float
        Net charge of the atom.
    pl : float
        Path length along the line of sight.
    gaunt_fun : function
        Function used to compute the gaunt
        free-free factor.

    Returns
    -------
    """

    return kappa(nu, te, ne, ni, z=z, gaunt_fun=gaunt_fun) * pl
