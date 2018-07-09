"""
This file defines a set of parameters for source reconstruction, such as:

- Method
- SNR/regularization
"""

import sys, os
from os.path import dirname, abspath

def get_params():
    """
    Returns a list of dictionaries. Each diciionary contains the fields:
        - 'method'
        - 'lambda'
        - 'snr'

    Note:
        lambda = 1 / snr**2
    """
    params = []

    index = 0
    #
    p = {}
    p['index'] = index
    index += 1
    p['method'] = 'dSPM'
    p['snr']    = 3.0
    p['lambda'] = 1.0 / (p['snr']**2)

    params.append(p)

    #
    p = {}
    p['index'] = index
    index += 1
    p['method'] = 'dSPM'
    p['snr']    = 1.0
    p['lambda'] = 1.0 / (p['snr']**2)
    params.append(p)


    #
    return params
