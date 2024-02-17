"""
Core functions for a generic radio recombination line.
"""

import numpy as np

from rrlpy.rrl.constants import Ry, k_B, h, m_e, e, c


def beta(bn, bm, nu, te):
    """
    Correction factor for stimulated emission/absorption.
    Eq. (31) Salgado et al. (2017b)

    Parameters
    ----------
    bn : float
        Departure coefficient for the lower level.
    bm : float
        Departure coefficient for the upper level.
    nu : float
        Frequency of the transition between levels m and n.
    te : float
        Electron temperature.

    Returns
    -------
    beta : float
        Correction factor for the transition between levels m and n.
    """

    exparg = -h * nu / (k_B * te)
    exp = np.exp(exparg)
    return (1.0 - bm / bn * exp) / (1.0 - exp)


def fnnp_app(n, dn):
    """
    Eq. (1) Menzel (1969)

    Parameters
    ----------
    n : int
        Principal quantum number.
    dn : int
        Jump between principal quantum numbers.

    Returns
    -------
    fnnp : float
        fnnp
    """

    return n * mdn(dn) * (1.0 + 1.5 * dn / n)


def mdn(dn):
    """
    Gives the :math:`M(\\Delta n)` factor for a given :math:`\\Delta n`.
    ref. Menzel (1968)

    Parameters
    ----------
    dn : int
        :math:`\\Delta n`. Up to n==5.

    Returns
    -------
    mdn : float
        :math:`M(\\Delta n)`

    :Example:

    >>> mdn(1)
    0.1908
    >>> mdn(5)
    0.001812
    """

    if dn == 1:
        mdn_ = 0.1908
    if dn == 2:
        mdn_ = 0.02633
    if dn == 3:
        mdn_ = 0.008106
    if dn == 4:
        mdn_ = 0.003492
    if dn == 5:
        mdn_ = 0.001812

    return mdn_


def tau_constant():
    """
    Constants that go into the RRL optical depth.
    """

    return (
        h**3 * e**2.0 * np.pi / (np.power(2.0 * np.pi * m_e * k_B, 3.0 / 2.0) * m_e * c)
    ).cgs


def tau_exact(n, ne, te, ni, pl, fnnp, nu, dn, z):
    """
    Optical depth of a RRL.

    Parameters
    ----------
    n : int
        Principal quantum number.
    ne : float
        Electron density.
    te : float
        Electron temperature.
    ni : float
        Collisional partner density.
    pl : float
        Path length along the line of sight.
    fnnp : float

    nu : float
        Frequency of the transition.
    dn : int
        Jump between energy levels.
    z : int
        Net charge of the atom.
    """

    cte = tau_constant()
    xi_ = xi(n, te, z)

    return (
        cte
        * n**2
        * fnnp
        * ne
        * ni
        * pl
        * np.power(te, -3.0 / 2.0)
        * np.exp(xi_)
        * (1.0 - np.exp(-h * nu / (k_B * te)))
    )


def xi(n, te, z):
    """
    Argument of the exponential factor in the Saha-Boltzmann equation.

    Parameters
    ----------
    n : int
        Principal quantum number.
    te : float
        Electron temperature in K.
    z : float
        Net charge of the ion.

    Returns
    -------
    xi : float
        :math:`z^2 Ry / (n^2 k_{B} te)`
    """

    return (z**2.0 * Ry / (n**2.0 * k_B * te)).cgs
