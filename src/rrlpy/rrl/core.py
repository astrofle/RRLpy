"""
Core functions for a generic radio recombination line.
"""

from rrlpy.rrl.constants import Ry, k_B


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

    return z**2.0 * Ry / (n**2.0 * k_B * te)
