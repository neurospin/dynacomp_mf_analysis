"""
Analyze correlation between (EOG decrease in self-similarity) and
(decrease of self-similarity of cortical region i) for i in range(138)
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

# select cumulant for analysis: 0 for H, 1 for M
cumulant_idx = 0

# Select groups and subjects
groups = ['AV', 'V', 'AVr']
subjects = info_sensor['subjects']
n_subjects = 36


# Scales for EOG
SCALE_1 = 10
SCALE_2 = 14   # fs = 2000, f1 = 0.1, f2 = 1.5


# Cortical data: with or without ICA preprocessing?
with_ica = True
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
mf_params_idx = 3

# Select source reconstruction parameters
source_rec_params_idx = 0

#------------------------------------------------------------------
# Functions
#------------------------------------------------------------------
def get_eog_log_cumulants(rest_condition, 
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
    all_log_cumulants_rest = np.zeros((n_subjects, n_channels, n_cumul))
    all_log_cumulants_task = np.zeros((n_subjects, n_channels, n_cumul))

    log2_e  = np.log2(np.exp(1))
    for ss in range(n_subjects):
        for nn in range(n_channels):
            for cc in range(n_cumul):
                c2j_rest = all_cumulants_rest[ss, nn, cc, :]
                c2j_task = all_cumulants_task[ss, nn, cc, :]

                x_reg       = np.arange(SCALE_1, SCALE_2+1)
                y_reg_rest  = c2j_rest[SCALE_1-1:SCALE_2]
                y_reg_task  = c2j_task[SCALE_1-1:SCALE_2]

                slope_rest, _, _, _, _ = linregress(x_reg,y_reg_rest)
                slope_task, _, _, _, _ = linregress(x_reg,y_reg_task)

                all_log_cumulants_rest[ss, nn, cc] = log2_e*slope_rest
                all_log_cumulants_task[ss, nn, cc] = log2_e*slope_task

    return all_log_cumulants_rest, all_log_cumulants_task



def get_cortex_log_cumulants(rest_condition, 
                             task_condition,
                             groups,
                             subjects,
                             mf_params_idx,
                             source_rec_params_idx):
    all_log_cumulants_rest,_ ,subjects_list = \
    mfr_source.load_data_groups_subjects(rest_condition, groups, subjects,
                                  mf_param_idx = mf_params_idx, 
                                  source_rec_param_idx = source_rec_params_idx,
                                  time_str = TIMESTR,
                                  extra_info = EXTRA_INFO)

    all_log_cumulants_task,_ ,subjects_list = \
        mfr_source.load_data_groups_subjects(task_condition, groups, subjects,
                                      mf_param_idx = mf_params_idx, 
                                      source_rec_param_idx = source_rec_params_idx,
                                      time_str = TIMESTR,
                                      extra_info = EXTRA_INFO)

    return all_log_cumulants_rest, all_log_cumulants_task

#------------------------------------------------------------------
# Load data
#------------------------------------------------------------------
eog_all_log_cumulants_rest, eog_all_log_cumulants_task = \
        get_eog_log_cumulants(rest_condition, 
                              task_condition,
                              groups,
                              subjects,
                              mf_params_idx)

eog_logcumul_rest = eog_all_log_cumulants_rest[:, :, cumulant_idx]
eog_logcumul_task = eog_all_log_cumulants_task[:, :, cumulant_idx]
eog_logcumul_diff = eog_logcumul_rest - eog_logcumul_task # shape (36, 2)



cortex_all_log_cumulants_rest, cortex_all_log_cumulants_task = \
        get_cortex_log_cumulants(rest_condition, 
                                 task_condition,
                                 groups,
                                 subjects,
                                 mf_params_idx,
                                 source_rec_params_idx)

cortex_logcumul_rest = cortex_all_log_cumulants_rest[:, :, cumulant_idx]
cortex_logcumul_task = cortex_all_log_cumulants_task[:, :, cumulant_idx] # shape (36, 138)
cortex_logcumul_diff = cortex_logcumul_rest - cortex_logcumul_task # shape (36, 138)


n_eog_channels   = 2
n_cortex_regions = 138

correlations = np.zeros((n_eog_channels, n_cortex_regions))
pvalues      = np.zeros((n_eog_channels, n_cortex_regions))


# Individual correlation
for eog_channel in range(n_eog_channels):
    diff_eog = eog_logcumul_diff[:, eog_channel]
    for cortex_region in range(n_cortex_regions):
        diff_cortex = cortex_logcumul_diff[:, cortex_region]

        corr, pval = pearsonr(diff_eog, diff_cortex)

        correlations[eog_channel, cortex_region] = corr
        pvalues[eog_channel, cortex_region]      = pval

# Apply FDR correction (separetely for each EOG channel)
pvalues[0, :] = multipletests(pvalues[0, :], alpha, method = 'fdr_bh')[1]
pvalues[1, :] = multipletests(pvalues[1, :], alpha, method = 'fdr_bh')[1]


# Set non significant correlations to zero
correlations[0, pvalues[0, :]>alpha] = 0.0
correlations[1, pvalues[1, :]>alpha] = 0.0


# Correlation between mean differences (average across channels)
eog_logcumul_diff_mean    = eog_logcumul_diff.mean(axis = 1)
cortex_logcumul_diff_mean = cortex_logcumul_diff.mean(axis = 1)

corr_mean, pval_mean = pearsonr(eog_logcumul_diff_mean, cortex_logcumul_diff_mean) 
corr_mean_2, pval_mean_2 = spearmanr(eog_logcumul_diff_mean, cortex_logcumul_diff_mean) 


print("Correlation between \
(mean eog diff) and (mean cortex diff) = %0.5f, pvalue = %0.5f"%(corr_mean, pval_mean))


# Plot correlations
file_1 = os.path.join('output_images','correlation_%seog_ch1_mf_%d.png'%(EXTRA_INFO, mf_params_idx))
file_2 = os.path.join('output_images','correlation_%seog_ch2_mf_%d.png'%(EXTRA_INFO, mf_params_idx))

plots.plot_brain(correlations[0, :], fmin = 0.00, fmax = 0.8, 
                    png_filename = file_1, positive_only = True)
plots.plot_brain(correlations[1, :], fmin = 0.00, fmax = 0.8, 
                    png_filename = file_2, positive_only = True)