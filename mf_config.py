"""
This file contains information about multifractal analysis parameters.
"""

import sys, os
from os.path import dirname, abspath
import numpy as np

def get_mf_params():
    """
    Returns a list of dictionaries. Each diciionary contains the fields,
    corresponding to parameters of MF analysis:
        - 'wt_name'
        - 'formalism' # multifractal formalism
        - 'p'         # value of p for p-Leaders
        - 'j1'
        - 'j2'
        - 'n_cumul'
        - 'gamint'
        - 'wtype'


    """
    mf_params = []

    # new param, wcmf
    param = {}
    param['wt_name']   = 'db3'
    param['formalism'] = 'wcmf'
    param['p']         = None
    param['j1']        = 8
    param['j2']        = 12
    param['n_cumul']   = 3
    param['gamint']    = 1.0
    param['wtype']     = 0
    mf_params.append(param)


    # new param - wlmf
    param = {}
    param['wt_name']   = 'db3'
    param['formalism'] = None
    param['p']         = np.inf
    param['j1']        = 8
    param['j2']        = 12
    param['n_cumul']   = 3
    param['gamint']    = 1.0
    param['wtype']     = 0
    mf_params.append(param)


    # new param - p = 1
    param = {}
    param['wt_name']   = 'db3'
    param['formalism'] = None
    param['p']         = 1.0
    param['j1']        = 8
    param['j2']        = 12
    param['n_cumul']   = 3
    param['gamint']    = 1.0
    param['wtype']     = 0
    mf_params.append(param)


    # new param - p = 2
    param = {}
    param['wt_name']   = 'db3'
    param['formalism'] = None
    param['p']         = 2.0
    param['j1']        = 8
    param['j2']        = 12
    param['n_cumul']   = 3
    param['gamint']    = 1.0
    param['wtype']     = 0
    mf_params.append(param)



    return mf_params
