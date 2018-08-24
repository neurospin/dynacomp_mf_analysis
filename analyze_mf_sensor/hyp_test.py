import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import numpy as np
import mfanalysis as mf
import matplotlib
import matplotlib.pyplot as plt
import os.path as op
import mne
import os
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

from statsmodels.stats.multitest import multipletests
import visualization_utils as v_utils
import sensor_mf_results as mfr

matplotlib.rcParams.update({'errorbar.capsize': 2})

#===============================================================================
# Global parameters 
#===============================================================================
# Load info
import meg_info_sensor_space
info   = meg_info_sensor_space.get_info()

#raw filename - only one raw file is necessary to get information about 
# sensor loacation -> used to plot
raw_filename = 'C:\\Users\\omard\\Documents\\data_to_go\\dynacomp\\AV_rest0_raw_trans_sss.fif'

# select sensor type
sensor_type = 'mag'


# Select groups and subjects
groups = ['AV', 'V' ,'AVr']
subjects = info['subjects']

# Select conditions
rest_condition = 'rest5'
task_condition = 'posttest'

# Select MF parameters and source reconstruction parameters
mf_params_idx = 1

# Load cumulants and log-cumulants
all_log_cumulants_rest, all_cumulants_rest, subjects_list, params, _, _, _ = \
     mfr.load_data_groups_subjects(rest_condition, mf_param_idx = mf_params_idx, channel_type = sensor_type)
all_log_cumulants_task, all_cumulants_task, subjects_list, params, channels_picks, channels_names, ch_name2index = \
    mfr.load_data_groups_subjects(task_condition, mf_param_idx = mf_params_idx, channel_type = sensor_type)

n_channels = len(channels_picks)

#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------
# Load raw to get info about sensor positions
raw = mne.io.read_raw_fif(raw_filename)

# get sensor positions via layout
pos = mne.find_layout(raw.info).pos[channels_picks, :]


correction_multiple_tests = 'fdr' # 'fdr', 'bonferroni' or None
alpha = 0.05

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def one_sided_ttest_rel(x,y):
    """
    Returns the value of the statistic and p-value for the paired Student t test
    with the hypotheses:
        H_0:  x = y
        H_1:  x < y

    For wilcoxon test, see::
        https://stackoverflow.com/questions/16296225/one-sided-wilcoxon-signed-rank-test-using-scipy
    """
    stat, pval = ttest_rel(x,y) # two-sided p-value, we need the one-sided!
    T = stat

    if T > 0: # x > y
        pval = 1.0-pval/2.0
    else:
        pval = pval/2.0

    return stat, pval

def pvals_correction(pvals):
    if correction_multiple_tests == 'fdr':
        # - Benjamini/Hochberg  (non-negative) =  'indep' in mne.fdr_correction
        _, pvals, _, _ = multipletests(pvals, alpha, method = 'fdr_bh')

    elif correction_multiple_tests == 'bonferroni':
                # bonferroni
        _, pvals, _, _ = multipletests(pvals, alpha, method = 'bonferroni')

    return pvals


#-------------------------------------------------------------------------------
# Test whether H_task - H_rest < 0 for each sensor
# Hypotheses:
# H_0:  H_task - H_rest = 0
# H_1:  H_task - H_rest < 0
#-------------------------------------------------------------------------------



H_rest = all_log_cumulants_rest[:, :, 0] # shape (n_subjects, n_sensors)
H_task = all_log_cumulants_task[:, :, 0] # shape (n_subjects, n_sensors)

H_pvals = np.ones(n_channels)

for ii in range(n_channels):
    H_rest_ii = H_rest[:, ii]
    H_task_ii = H_task[:, ii]
    stat, pval = one_sided_ttest_rel(H_task_ii, H_rest_ii)

    H_pvals[ii] = pval


# correction for multiple comparisons
H_pvals = pvals_correction(H_pvals)
H_signif = H_pvals < alpha


# Plot significant differences:
H_diff = -(H_task.mean(axis=0) - H_rest.mean(axis=0))
H_diff[~H_signif] = 0.0
v_utils.plot_data_topo(H_diff, pos, title = '(H_rest - H_task) tested for H_task < H_rest', cmap = 'Reds')

#-------------------------------------------------------------------------------
# Test whether avg(C2(j)_task) - avg(C2(j)_rest) < 0 for each sensor
#-------------------------------------------------------------------------------

# compute averages across subjects
avg_C2j_rest = all_cumulants_rest[:,:,:,9:14].max(axis = 3)
avg_C2j_rest = avg_C2j_rest[:,:,1]   # shape (n_subjects, n_sensors)
avg_C2j_task = all_cumulants_task[:,:,:,9:14].max(axis = 3)
avg_C2j_task = avg_C2j_task[:,:,1]   # shape (n_subjects, n_sensors)


avgC2j_pvals = np.ones(n_channels)

for ii in range(n_channels):
    avgC2j_rest_ii = avg_C2j_rest[:, ii]
    avgC2j_task_ii = avg_C2j_task[:, ii]
    stat, pval = one_sided_ttest_rel(avgC2j_task_ii, avgC2j_rest_ii)

    avgC2j_pvals[ii] = pval


# correction for multiple comparisons
avgC2j_pvals = pvals_correction(avgC2j_pvals)
avgC2j_signif = avgC2j_pvals < alpha


# Plot significant differences:
avgC2j_diff = -(avg_C2j_task.mean(axis=0) - avg_C2j_rest.mean(axis=0))
avgC2j_diff[~avgC2j_signif] = 0.0
v_utils.plot_data_topo(avgC2j_diff, pos, title = '(avg_C2j_rest - avg_C2j_task) tested for avg_C2j_task < avg_C2j_rest', cmap = 'Reds')


plt.show()
