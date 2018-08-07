"""
Analyze correlation between (EOG decrease of maxminC2j) and
(decrease of maxminC2j of sensor i) for i in range(n_channels)
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

import mne

import visualization_utils as v_utils

#------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------

# Load info
import meg_info_sensor_space
import meg_info

info_sensor   = meg_info_sensor_space.get_info()
info_source   = meg_info.get_info()


#raw filename - only one raw file is necessary to get information about 
# sensor loacation -> used to plot
raw_filename = '/neurospin/tmp/Omar/AV_rest0_raw_trans_sss.fif'

# alpha for hyp testing
alpha = 0.05


# Select groups and subjects
groups = ['AV', 'V', 'AVr']
subjects = info_sensor['subjects']
n_subjects = 36


# Scales for EOG
SCALE_1_eog = 10
SCALE_2_eog = 14   # fs = 2000, f1 = 0.1, f2 = 1.5


# Scales for MAG
SCALE_1_mag = 9
SCALE_2_mag = 13   # fs = 2000, f1 = 0.1, f2 = 1.5


# Select conditions
rest_condition = 'rest5'
task_condition = 'posttest'

# Select MF parameters
mf_params_idx = 1


#------------------------------------------------------------------
# Functions
#------------------------------------------------------------------
def get_eog_maxC2j(rest_condition, 
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
    maxC2j_rest = all_cumulants_rest[:,:,1, SCALE_1_eog-1:SCALE_2_eog].max(axis=2)
    maxC2j_task = all_cumulants_task[:,:,1, SCALE_1_eog-1:SCALE_2_eog].max(axis=2)

    return maxC2j_rest, maxC2j_task



def get_mag_maxC2j(rest_condition, 
                          task_condition,
                          groups,
                          subjects,
                          mf_params_idx):

     # Load cumulants and log-cumulants
    all_log_cumulants_rest, all_cumulants_rest, subjects_list, params, _, _, _ = \
         mfr_sensor.load_data_groups_subjects(rest_condition, 
                                       groups = groups,
                                       subjects = subjects,
                                       mf_param_idx = mf_params_idx, 
                                       channel_type = 'mag')
    all_log_cumulants_task, all_cumulants_task, subjects_list, params, channels_picks, channels_names, ch_name2index = \
        mfr_sensor.load_data_groups_subjects(task_condition, 
                                      groups = groups,
                                      subjects = subjects, 
                                      mf_param_idx = mf_params_idx, 
                                      channel_type = 'mag')

    maxC2j_rest = all_cumulants_rest[:,:,1, SCALE_1_eog-1:SCALE_2_mag].max(axis=2)
    maxC2j_task = all_cumulants_task[:,:,1, SCALE_1_eog-1:SCALE_2_mag].max(axis=2)

    return maxC2j_rest, maxC2j_task, channels_picks
#------------------------------------------------------------------
# Load data
#------------------------------------------------------------------
eog_maxC2j_rest, eog_maxC2j_task = \
        get_eog_maxC2j(rest_condition, 
                              task_condition,
                              groups,
                              subjects,
                              mf_params_idx)

eog_maxC2j_diff = eog_maxC2j_rest - eog_maxC2j_task # shape (36, 2)



mag_maxC2j_rest, mag_maxC2j_task, mag_channels_picks = \
        get_mag_maxC2j(rest_condition, 
                              task_condition,
                              groups,
                              subjects,
                              mf_params_idx)

mag_maxC2j_diff = mag_maxC2j_rest - mag_maxC2j_task # shape (36, 102)


n_eog_channels  = 2
n_mag_channels  = 102

correlations = np.zeros((n_eog_channels, n_mag_channels))
pvalues      = np.zeros((n_eog_channels, n_mag_channels))


# Individual correlation
for eog_channel in range(n_eog_channels):
    diff_eog = eog_maxC2j_diff[:, eog_channel]
    for mag_region in range(n_mag_channels):
        diff_mag = mag_maxC2j_diff[:, mag_region]

        corr, pval = pearsonr(diff_eog, diff_mag)

        correlations[eog_channel, mag_region] = corr
        pvalues[eog_channel, mag_region]      = pval

# Apply FDR correction (separetely for each EOG channel)
pvalues[0, :] = multipletests(pvalues[0, :], alpha, method = 'fdr_bh')[1]
pvalues[1, :] = multipletests(pvalues[1, :], alpha, method = 'fdr_bh')[1]


# Set non significant correlations to zero
correlations[0, pvalues[0, :]>alpha] = 0.0
correlations[1, pvalues[1, :]>alpha] = 0.0


# Correlation between mean differences (average across channels)
eog_maxC2j_diff_mean    = eog_maxC2j_diff.mean(axis = 1)
mag_maxC2j_diff_mean = mag_maxC2j_diff.mean(axis = 1)

corr_mean, pval_mean = pearsonr(eog_maxC2j_diff_mean, mag_maxC2j_diff_mean) 
corr_mean_2, pval_mean_2 = spearmanr(eog_maxC2j_diff_mean, mag_maxC2j_diff_mean) 


print("Correlation between \
(mean eog diff) and (mean mag diff) = %0.5f, pvalue = %0.5f"%(corr_mean, pval_mean))


# Plot correlation

# Load raw to get info about sensor positions
raw = mne.io.read_raw_fif(raw_filename)
# get sensor positions via layout
pos = mne.find_layout(raw.info).pos[mag_channels_picks, :]
v_utils.plot_data_topo(correlations[0, :], pos, vmin = 0.0, vmax = 0.8, title = 'correlations for channel 1', cmap = 'Reds')
v_utils.plot_data_topo(correlations[1, :], pos, vmin = 0.0, vmax = 0.8, title = 'correlations for channel 2', cmap = 'Reds')

plt.show()