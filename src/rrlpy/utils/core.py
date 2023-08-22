""" Utility functions for RRLpy """


import numpy as np


def fwhm2sigma(fwhm):
    """
    Converts a FWHM to the standard deviation, :math:`\\sigma` of a Gaussian distribution.

    .. math:

       FWHM=2\\sqrt{2\\ln2}\\sigma

    Parameters
    ----------
    fwhm : float
        Full Width at Half Maximum of the Gaussian.

    Returns
    -------
    sigma : float
        Equivalent standard deviation of a Gausian with a Full Width at Half Maximum `fwhm`.

    :Example:

    >>> 1/fwhm2sigma(1)
    2.3548200450309493
    """

    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gauss_area(amplitude, sigma):
    """
    Returns the area under a Gaussian of a given amplitude and sigma.

    .. math:

        Area=\\sqrt(2\\pi)A\\sigma

    Parameters
    ----------
    amplitude : float
        Amplitude of the Gaussian, :math:`A`.
    sigma : float
        Standard deviation fo the Gaussian, :math:`\\sigma`.

    Returns
    -------
    area : float
        The area under a Gaussian of a given amplitude and standard deviation.
    """

    return amplitude * sigma * np.sqrt(2.0 * np.pi)


def gauss_area_err(amplitude, amplitude_err, sigma, sigma_err):
    """
    Returns the error on the area of a Gaussian of a given `amplitude` and `sigma` \
    with their corresponding errors. It assumes no correlation between `amplitude` and
    `sigma`.

    Parameters
    ----------
    amplitude : float
        Amplitude of the Gaussian.
    amplitude_err : float
        Error on the amplitude.
    sigma : float
        Standard deviation of the Gaussian.
    sigma_err : float
        Error on sigma.

    Returns
    -------
    area_err : float
        The error on the area.
    """

    err1 = np.power(amplitude_err * sigma * np.sqrt(2 * np.pi), 2)
    err2 = np.power(sigma_err * amplitude * np.sqrt(2 * np.pi), 2)

    return np.sqrt(err1 + err2)
