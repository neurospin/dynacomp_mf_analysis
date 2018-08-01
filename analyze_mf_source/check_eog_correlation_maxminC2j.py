"""
Analyze correlation between (EOG (maxC2j-minC2j)_rest - (maxC2j-minC2j)_task ) and
(maxC2j-minC2j)_rest - (maxC2j-minC2j)_task  of cortical region i) for i in range(138)
"""

import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import numpy as np
import sensor_mf_results as mfr_sensor
import source_mf_results as mfr_source
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

import plots_ as plots

import mne


#------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------

# Load info
import meg_info_sensor_space
import meg_info

info_sensor   = meg_info_sensor_space.get_info()
info_source   = meg_info.get_info()


# alpha for hyp testing
alpha = 0.05


# Select groups and subjects
groups = ['AV', 'V', 'AVr']
subjects = info_sensor['subjects']
n_subjects = 36


# Scales for EOG
SCALE_1_eog = 10
SCALE_2_eog = 14   # fs = 2000, f1 = 0.1, f2 = 1.5

# Scales for mag
SCALE_1_cortex = 8
SCALE_2_cortex = 12   # fs = 400, f1 = 0.1, f2 = 1.5

# Cortical data: with or without ICA preprocessing?
with_ica = False
if with_ica:
    EXTRA_INFO =  ''
    TIMESTR    =  '20180713'
else:
    EXTRA_INFO = 'no_ica_'  # ''
    TIMESTR    = '20180724' # '20180713'


# Select conditions
rest_condition = 'rest5'
task_condition = 'posttest'

# Select MF parameters
mf_params_idx = 1

# Select source reconstruction parameters
source_rec_params_idx = 0

#------------------------------------------------------------------
# Functions
#------------------------------------------------------------------
def get_eog_maxminC2j(rest_condition, 
                          task_condition,
                          groups,
                          subjects,
                          mf_params_idx):
    # Load cumulants and log-cumulants
    _, all_cumulants_rest, subjects_list, params, _, _, _ = \
         mfr_sensor.load_data_groups_subjects(rest_condition, 
                                       groups = groups,
                                       subjects = subjects,
                                       mf_param_idx = mf_params_idx, 
                                       channel_type = 'EOG')
    _, all_cumulants_task, subjects_list, params, channels_picks, channels_names, ch_name2index = \
        mfr_sensor.load_data_groups_subjects(task_condition, 
                                      groups = groups,
                                      subjects = subjects, 
                                      mf_param_idx = mf_params_idx, 
                                      channel_type = 'EOG')

    n_subjects = all_cumulants_rest.shape[0]
    n_channels = all_cumulants_rest.shape[1]
    n_cumul    = all_cumulants_rest.shape[2]

    maxminC2j_rest = (all_cumulants_rest[:, :, 1, SCALE_1_eog-1:SCALE_2_eog]).max(axis = 2) \
                     - (all_cumulants_rest[:, :, 1, SCALE_1_eog-1:SCALE_2_eog]).min(axis = 2)

    maxminC2j_task = (all_cumulants_task[:, :, 1, SCALE_1_eog-1:SCALE_2_eog]).max(axis = 2) \
                     - (all_cumulants_task[:, :, 1, SCALE_1_eog-1:SCALE_2_eog]).min(axis = 2)

    return maxminC2j_rest, maxminC2j_task



def get_cortex_maxminC2j(rest_condition, 
                         task_condition,
                         groups,
                         subjects,
                         mf_params_idx,
                         source_rec_params_idx):

    _,all_cumulants_rest ,subjects_list = \
    mfr_source.load_data_groups_subjects(rest_condition, groups, subjects,
                                  mf_param_idx = mf_params_idx, 
                                  source_rec_param_idx = source_rec_params_idx,
                                  time_str = TIMESTR,
                                  extra_info = EXTRA_INFO)

    _,all_cumulants_task ,subjects_list = \
        mfr_source.load_data_groups_subjects(task_condition, groups, subjects,
                                      mf_param_idx = mf_params_idx, 
                                      source_rec_param_idx = source_rec_params_idx,
                                      time_str = TIMESTR,
                                      extra_info = EXTRA_INFO)


    maxminC2j_rest = (all_cumulants_rest[:, :, 1, SCALE_1_cortex-1:SCALE_2_cortex]).max(axis = 2) \
                     - (all_cumulants_rest[:, :, 1, SCALE_1_cortex-1:SCALE_2_cortex]).min(axis = 2)

    maxminC2j_task = (all_cumulants_task[:, :, 1, SCALE_1_cortex-1:SCALE_2_cortex]).max(axis = 2) \
                     - (all_cumulants_task[:, :, 1, SCALE_1_cortex-1:SCALE_2_cortex]).min(axis = 2)
                     
    return maxminC2j_rest, maxminC2j_task

#------------------------------------------------------------------
# Load data
#------------------------------------------------------------------
eog_maxminC2j_rest, eog_maxminC2j_task = \
                get_eog_maxminC2j(rest_condition, 
                                      task_condition,
                                      groups,
                                      subjects,
                                      mf_params_idx)

eog_maxminC2j_diff = eog_maxminC2j_rest - eog_maxminC2j_task # shape (36, 2)



cortex_maxminC2j_rest, cortex_maxminC2j_task = \
                get_cortex_maxminC2j(rest_condition, 
                                         task_condition,
                                         groups,
                                         subjects,
                                         mf_params_idx,
                                         source_rec_params_idx)

cortex_maxminC2j_diff = cortex_maxminC2j_rest - cortex_maxminC2j_task # shape (36, 138)


n_eog_channels   = 2
n_cortex_regions = 138

correlations = np.zeros((n_eog_channels, n_cortex_regions))
pvalues      = np.zeros((n_eog_channels, n_cortex_regions))


# Individual correlation
for eog_channel in range(n_eog_channels):
    diff_eog = eog_maxminC2j_diff[:, eog_channel]
    for cortex_region in range(n_cortex_regions):
        diff_cortex = cortex_maxminC2j_diff[:, cortex_region]

        corr, pval = spearmanr(diff_eog, diff_cortex)

        correlations[eog_channel, cortex_region] = corr
        pvalues[eog_channel, cortex_region]      = pval

# Apply FDR correction (separetely for each EOG channel)
pvalues[0, :] = multipletests(pvalues[0, :], alpha, method = 'fdr_bh')[1]
pvalues[1, :] = multipletests(pvalues[1, :], alpha, method = 'fdr_bh')[1]


# Set non significant correlations to zero
correlations[0, pvalues[0, :]>alpha] = 0.0
correlations[1, pvalues[1, :]>alpha] = 0.0


# Correlation between mean differences (average across channels)
eog_maxminC2j_diff_mean    = eog_maxminC2j_diff.mean(axis = 1)
cortex_maxminC2j_diff_mean = cortex_maxminC2j_diff.mean(axis = 1)

corr_mean, pval_mean = pearsonr(eog_maxminC2j_diff_mean, cortex_maxminC2j_diff_mean) 
corr_mean_2, pval_mean_2 = spearmanr(eog_maxminC2j_diff_mean, cortex_maxminC2j_diff_mean) 


print("Correlation between \
(mean eog diff) and (mean cortex diff) = %0.5f, pvalue = %0.5f"%(corr_mean_2, pval_mean_2))


# Plot correlation
file_1 = os.path.join('output_images','maxminC2j_correlation_%seog_ch1_mf_%d.png'%(EXTRA_INFO, mf_params_idx))
file_2 = os.path.join('output_images','maxminC2j_correlation_%seog_ch2_mf_%d.png'%(EXTRA_INFO, mf_params_idx))



plots.plot_brain(correlations[0, :], fmin = 0.00, fmax = 0.8, 
                    png_filename = file_1, positive_only = True)
plots.plot_brain(correlations[1, :], fmin = 0.00, fmax = 0.8, 
                    png_filename = file_2, positive_only = True)


