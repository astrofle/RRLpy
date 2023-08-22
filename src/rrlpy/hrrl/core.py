"""Core functions for hydrogen radio recombination lines"""


import numpy as np


def line_brightness(te, tbg0, tm, tau_c, tau_l_lte, tau_l, bm):
    """
    HRRL brightness following the model of Shaver (1975).

    """

    fac1 = tbg0 * np.exp(-tau_c) * (np.exp(-tau_l) - 1.0)
    fac2 = te * (
        (bm * tau_l_lte + tau_c) / (tau_l + tau_c) * (1.0 - np.exp(-(tau_l + tau_c)))
        - (1.0 - np.exp(-tau_c))
    )
    fac3 = tm * (
        (1.0 - np.exp(-(tau_l + tau_c))) / (tau_l + tau_c) - (1.0 - np.exp(-tau_c)) / tau_c
    )

    tlb = fac1 + fac2 + fac3

    return tlb
