import os
import numpy as np


def find_lines_sb(freq, line, z=0, verbose=False):
    """
    Finds if there are any lines of a given type in the frequency range.
    The line frequencies are corrected for redshift.

    :param freq: Frequency axis in which to search for lines (MHz). It should not contain \
    NaN or inf values.
    :type freq: array
    :param line: Line type to search for.
    :type line: string
    :param z: Redshift to apply to the rest frequencies.
    :type z: float
    :param verbose: Verbose output?
    :type verbose: bool
    :returns: Lists with the princpipal quantum number and the reference \
    frequency of the line. The frequencies are redshift corrected in MHz.
    :rtype: array.

    See Also
    --------
    load_ref : Describes the format of line and the available ones.

    Examples
    --------
    >>> from crrlpy import crrls
    >>> freq = [10, 11]
    >>> ns, rf = crrls.find_lines_sb(freq, 'RRL_CIalpha')
    >>> ns
    array([843., 844., 845., 846., 847., 848., 849., 850., 851., 852., 853.,
           854., 855., 856., 857., 858., 859., 860., 861., 862., 863., 864.,
           865., 866., 867., 868., 869.])
    """

    # Load the reference frequencies.
    qn, restfreq = load_ref(line)

    # Correct rest frequencies for redshift.
    reffreq = restfreq / (1.0 + z)

    if verbose:
        print("Subband edges: {0}--{1}".format(freq[0], freq[-1]))

    # Check which lines lie within the sub band.
    mask_ref = (freq[0] < reffreq) & (freq[-1] >= reffreq)
    reffreqs = reffreq[mask_ref]
    refqns = qn[mask_ref]

    nlin = len(reffreqs)
    if verbose:
        print("Found {0} {1} lines within the subband.".format(nlin, line))
        if nlin > 1:
            print("Corresponding to n values: {0}--{1}".format(refqns[0], refqns[-1]))
        elif nlin == 1:
            print(
                "Corresponding to n value {0} and frequency {1} MHz".format(refqns[0], reffreqs[0])
            )

    return refqns, reffreqs


def load_ref(line):
    """
    Loads the reference spectrum for the specified line.

    | Available lines:
    | RRL_CIalpha
    | RRL_CIbeta
    | RRL_CIdelta
    | RRL_CIgamma
    | RRL_CI13alpha
    | RRL_HeIalpha
    | RRL_HeIbeta
    | RRL_HIalpha
    | RRL_HIbeta
    | RRL_SIalpha
    | RRL_SIbeta

    More lines can be added by including a list in the
    linelist directory.

    Parameters
    ----------
    line : string
           Line for which the principal quantum number and reference frequencies are desired.

    Returns
    -------
    n : array
            Principal quantum numbers.
    reference_frequencies : array
                            Reference frequencies of the lines inside the spectrum in MHz.
    """

    LOCALDIR = os.path.dirname(os.path.realpath(__file__))
    refspec = np.loadtxt("{0}/data/{1}.txt".format(LOCALDIR, line), usecols=(2, 3))
    qn = refspec[:, 0]
    reffreq = refspec[:, 1]

    return qn, reffreq
