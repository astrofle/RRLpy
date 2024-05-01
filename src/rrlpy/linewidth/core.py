"""
Line width functions.
"""


import numpy as np
from astropy.constants import k_B
from scipy import interpolate

from rrlpy.utils import fwhm2sigma, sigma2fwhm


def doppler_broad(t, m, vrms, fwhm=False):
    """
    Doppler broadening.

    :math:`\\frac{\\Delta v}{\\mbox{m s}^{-1}}=(\\frac{2k_{B}T}{m}+v_{\\mathrm{rms}}^2)^{1/2}`

    :param t: Gas temperature in K.
    :type t: float
    :param m: Mass of the element producing the line in amu.
    :type m: float
    :param vrms: Turbulent velocity in :math:`\\mbox{m s}^{-1}`.
    :type vrms: float
    :returns: The sigma or FWHM of a Gaussian line due to Doppler broadening in :math:`\\mbox{m s}^{-1}`.
    :rtype: float
    """

    dv = np.sqrt(8314.46262103 * t / m + np.power(vrms, 2.0))

    if fwhm:
        dv = sigma2fwhm(dv)

    return dv


def doppler_temp(sigma, m, vrms, fwhm=False):
    """
    The temperature required to produce a Gaussian line of width sigma.

    :param sigma: The sigma or FWHM of a Gaussian line due to Doppler broadening in :math:`\\mbox{m s}^{-1}`.
    :type sigma: float
    :param m: Mass of the element producing the line in amu.
    :type m: float
    :param vrms: Turbulent velocity in :math:`\\mbox{m s}^{-1}`.
    :type vrms: float
    :returns: Gas temperature in K.
    :rtype: float
    """

    dv = sigma
    if fwhm:
        dv = fwhm2sigma(dv)

    return (np.power(dv, 2.0) - np.power(vrms, 2.0)) * m / 8314.46262103


def doppler_temp_err(sigma, sigma_err, m, vrms, vrms_err, fwhm=False):
    """
    Error on the temperature derived from the Doppler line width.

    Parameters
    ----------
    sigma : float
        Doppler width.
    sigma_err : float
        Error on the Doppler width.
    m : float
        Mass of the element producing the line in amu.
    vrms : float
        Turbulent velocity in :math:`\\mbox{m s}^{-1}`.
    vrms_err : float
        Error on the turbulent velocity in :math:`\\mbox{m s}^{-1}`.
    fwhm : bool
        False if the Doppler width is the standard deviation of a
        Gaussian. True if the Doppler width is the FWHM.

    Returns
    -------
    temp_err : float
        Error on the temperature in K.
    """

    dv = sigma
    dv_err = sigma_err
    if fwhm:
        dv = fwhm2sigma(dv)
        dv_err = fwhm2sigma(dv_err)

    cte = m / 8314.46262103
    fac1 = 2.0 * dv * dv_err
    fac2 = 2.0 * vrms * vrms_err

    return np.sqrt((np.power(fac1, 2.0) + np.power(fac2, 2.0))) * cte


def pressure_broad(n, te, ne):
    """
    Pressure induced broadening in Hz.
    Shaver (1975) Eq. (64a) for te <= 1000 K and
    Eq. (61) for te > 1000 K.

    Parameters
    ----------

    """

    if te <= 1000:
        dnup = 2e-5 * np.power(te, -3.0 / 2.0) * np.exp(-26.0 / np.power(te, 1.0 / 3.0)) * ne * np.power(n, 5.2)
    else:
        dnup = 3.74e-8 * ne * np.power(n, 4.4) * np.power(te, -0.1)

    return dnup


def pressure_broad_salgado(n, te, ne, dn=1):
    """
    Pressure induced broadening in Hz.
    This gives the FWHM of a Lorentzian line.
    Salgado et al. (2017)

    :param n: Principal quantum number for which to compute the line broadening.
    :type n: float or array
    :param Te: Electron temperature to use when computing the collisional line width.
    :type Te: float
    :param ne: Electron density to use when computing the collisional line width.
    :type ne: float
    :param dn: Difference between the upper and lower level for which the line width is computed. (default 1)
    :type dn: int
    :returns: The collisional broadening FWHM in Hz using Salgado et al. (2015) formulas.
    :rtype: float or array
    """

    a, g = pressure_broad_coefs(te)

    return ne * np.power(10.0, a) * (np.power(n, g) + np.power(n + dn, g)) / 2.0 / np.pi


def pressure_broad_coefs(t):
    """
    Defines the values of the constants :math:`a` and :math:`\\gamma` that go into the collisional broadening formula
    of Salgado et al. (2017).

    Parameters
    ----------
    t : float
        Electron temperature.

    Returns
    -------
    coefs : list
        The values of :math:`a` and :math:`\\gamma`.
    """

    te = [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
        20000,
        30000,
    ]

    a = [
        -10.974098,
        -10.669695,
        -10.494541,
        -10.370271,
        -10.273172,
        -10.191374,
        -10.124309,
        -10.064037,
        -10.010153,
        -9.9613006,
        -9.6200366,
        -9.4001678,
        -9.2336349,
        -9.0848840,
        -8.9690170,
        -8.8686695,
        -8.7802238,
        -8.7012421,
        -8.6299908,
        -8.2718376,
        -8.0093937,
        -7.8344941,
        -7.7083367,
        -7.6126791,
        -7.5375720,
        -7.4770500,
        -7.4272885,
        -7.3857095,
        -7.1811733,
        -7.1132522,
    ]

    gammac = [
        5.4821631,
        5.4354009,
        5.4071360,
        5.3861013,
        5.3689105,
        5.3535398,
        5.3409679,
        5.3290318,
        5.3180304,
        5.3077770,
        5.2283700,
        5.1700702,
        5.1224893,
        5.0770049,
        5.0408369,
        5.0086342,
        4.9796105,
        4.9532071,
        4.9290080,
        4.8063682,
        4.7057576,
        4.6356118,
        4.5831746,
        4.5421547,
        4.5090104,
        4.4815675,
        4.4584053,
        4.4385507,
        4.3290786,
        4.2814240,
    ]

    a_func = interpolate.interp1d(te, a, kind="linear", bounds_error=False, fill_value="extrapolate")

    g_func = interpolate.interp1d(te, gammac, kind="linear", bounds_error=False, fill_value="extrapolate")

    return [a_func(t), g_func(t)]


def radiation_broad(n, W, tr):
    """
    Radiation induced broadening in Hz.
    Shaver (1975)
    """

    return 8e-17 * W * tr * np.power(n, 5.8)


def radiation_broad_salgado(n, w, tr):
    """
    Radiation induced broadening in Hz.
    This gives the FWHM of a Lorentzian line.
    Salgado et al. (2017)
    """

    return 6.096e-17 * w * tr * np.power(n, 5.8)


def radiation_broad_salgado_general(n, w, tr, nu0, alpha):
    """
    Radiation induced broadening in Hz.
    This gives the FWHM of a Lorentzian line.
    The expression is valid for power law like radiation fields.
    Salgado et al. (2017)
    """

    cte = 2.0 / np.pi * 2.14e4 * np.power(6.578e15 / nu0, alpha + 1.0) * k_B.cgs.value * nu0
    dnexp = alpha - 2.0

    return w * cte * tr * np.power(n, -3.0 * alpha - 2.0) * (1.0 + np.power(2.0, dnexp) + np.power(3.0, dnexp))


def voigt_fwhm(dD, dL):
    """
    Computes the FWHM of a Voigt profile. \
    http://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile

    .. math::

        FWHM_{\\rm{V}}=0.5346dL+\\sqrt{0.2166dL^{2}+dD^{2}}

    Parameters
    ----------
    dD : float
        FWHM of the Gaussian core.
    dL: float
        FWHM of the Lorentz wings.

    Returns
    -------
    fwhm : float
         The FWHM of a Voigt profile.
    """

    return np.multiply(0.5346, dL) + np.sqrt(np.multiply(0.2166, np.power(dL, 2)) + np.power(dD, 2))
