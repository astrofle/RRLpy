""" Utility functions for RRLpy """


import numpy as np
from scipy.interpolate import interp1d


def best_match_indx(value, array):
    """
    Searchs for the index of the closest entry to value inside an array.

    Parameters
    ----------
    value : Float
        Value to find inside the array.
    array : array
        List to search for the given value.

    Returns
    -------
    index : int
        Best match index for the value inside array.

    Examples
    --------

    >>> a = [1,2,3,4]
    >>> best_match_indx(3, a)
    2

    """

    array = np.array(array)

    return np.argmin(abs(array - value))


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


def interpolate(x, y, xnew):
    """
    Interpolate data (`x`,`y`) into `xnew`.

    Parameters
    ----------
    x : array
    y : array

    Returns
    -------
    """

    mask = ~np.isfinite(y)

    ynew = np.zeros(len(xnew), dtype=y.dtype)

    # Mask non-finite values.
    np.ma.masked_where(mask, y)
    mx = np.ma.masked_where(mask, x)
    valid = np.ma.flatnotmasked_contiguous(mx)

    # Interpolate non masked ranges indepently.
    if not isinstance(valid, slice):
        for rng in valid:
            if len(x[rng]) > 1:
                interp_y = interp1d(x[rng], y[rng], kind="linear", bounds_error=False, fill_value=0.0)
                ynew += interp_y(xnew)
            elif not np.isnan(x[rng]):
                ynew[best_match_indx(x[rng], xnew)] += y[rng]
    else:
        interp_y = interp1d(x[valid], y[valid], kind="linear", bounds_error=False, fill_value=0.0)
        ynew += interp_y(xnew)

    ynew[ynew == 0] = np.nan

    return ynew


def sigma2fwhm(sigma):
    """
    Converts the :math:`\\sigma` parameter of a Gaussian distribution to its FWHM.

    .. math:

       FWHM=2\\sqrt{2\\ln2}\\sigma

    Parameters
    ----------
    sigma : float
        Standard deviation of a Gaussian.

    Returns
    -------
    fwhm : float
        Full Width at Half Maximum of the Gaussian.
    """

    return sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))
