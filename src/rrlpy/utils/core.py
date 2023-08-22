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
