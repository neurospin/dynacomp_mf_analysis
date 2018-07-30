import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import numpy as np
import sensor_mf_results as mfr
import matplotlib.pyplot as plt
from scipy.stats import linregress

import visualization_utils as v_utils
import mne

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, cross_validate


SCALE_1 = 10
SCALE_2 = 14   # fs = 2000, f1 = 0.1, f2 = 1.5

def get_scales(fs, min_f, max_f):
    """
    Compute scales corresponding to the analyzed frequencies
    """
    f0 = (3.0/4.0)*fs
    j1 = int(np.ceil(np.log2(f0/max_f)))
    j2 = int(np.ceil(np.log2(f0/min_f)))
    return j1, j2


#===============================================================================
# Global parameters 
#===============================================================================
# Load info
import meg_info_sensor_space
info   = meg_info_sensor_space.get_info()

#raw filename - only one raw file is necessary to get information about 
# sensor loacation -> used to plot
raw_filename = '/neurospin/tmp/Omar/AV_rest0_raw_trans_sss.fif'

# select sensor type
sensor_type = 'EOG'

# select analysis type: 0 for H, 1 for M
see_cumulant = 1

# Select groups and subjects
groups = ['AV', 'V', 'AVr']
subjects = {}
subjects = info['subjects']
n_subjects = 36


# Select conditions
rest_condition = 'rest5'
task_condition = 'posttest'

# Select MF parameters and source reconstruction parameters
mf_params_idx = 1

# Load cumulants and log-cumulants
_, all_cumulants_rest, subjects_list, params, _, _, _ = \
     mfr.load_data_groups_subjects(rest_condition, 
     							   groups = groups,
     							   subjects = subjects,
     							   mf_param_idx = mf_params_idx, 
     							   channel_type = sensor_type)
_, all_cumulants_task, subjects_list, params, channels_picks, channels_names, ch_name2index = \
    mfr.load_data_groups_subjects(task_condition, 
     							  groups = groups,
     							  subjects = subjects, 
    							  mf_param_idx = mf_params_idx, 
    							  channel_type = sensor_type)

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






#===============================================================================
# Visualization
#===============================================================================
H_rest = all_log_cumulants_rest[:, :, 0]
H_task = all_log_cumulants_task[:, :, 0]

plt.figure()
plt.plot(H_rest[:, 0], H_rest[:, 1], 'bo', label=rest_condition)
plt.plot(H_task[:, 0], H_task[:, 1], 'ro', label=task_condition)
plt.legend()
plt.grid()

plt.figure()
plt.plot(H_rest[:, 0], 'bo-', label=rest_condition+' EOG channel 1')
plt.plot(H_task[:, 0], 'ro-', label=task_condition+' EOG channel 1')
plt.xlabel('subject')
plt.ylabel('H')
plt.legend()
plt.grid()

plt.figure()
plt.plot(H_rest[:, 1], 'bo-', label=rest_condition+' EOG channel 2')
plt.plot(H_task[:, 1], 'ro-', label=task_condition+' EOG channel 2')
plt.xlabel('subject')
plt.ylabel('H')
plt.legend()
plt.grid()


C1j_rest_ch1 = (all_cumulants_rest[:, 0, 0, :]).mean(axis = 0)
C1j_rest_ch2 = (all_cumulants_rest[:, 1, 0, :]).mean(axis = 0)

C1j_task_ch1 = (all_cumulants_task[:, 0, 0, :]).mean(axis = 0)
C1j_task_ch2 = (all_cumulants_task[:, 1, 0, :]).mean(axis = 0)



C2j_rest_ch1 = (all_cumulants_rest[:, 0, 1, :]).mean(axis = 0)
C2j_rest_ch2 = (all_cumulants_rest[:, 1, 1, :]).mean(axis = 0)

C2j_task_ch1 = (all_cumulants_task[:, 0, 1, :]).mean(axis = 0)
C2j_task_ch2 = (all_cumulants_task[:, 1, 1, :]).mean(axis = 0)



v_utils.plot_cumulants([C1j_rest_ch1, C1j_task_ch1], 
                        j1=10, j2=14, 
                        title = 'C1(j) EOG channel 1', 
                        labels = [rest_condition, task_condition])

v_utils.plot_cumulants([C1j_rest_ch2, C1j_task_ch2], 
                        j1=10, j2=14, 
                        title = 'C1(j) EOG channel 2', 
                        labels = [rest_condition, task_condition])


v_utils.plot_cumulants([C2j_rest_ch1, C2j_task_ch1], 
                        j1=10, j2=14, 
                        title = 'C2(j) EOG channel 1', 
                        labels = [rest_condition, task_condition])

v_utils.plot_cumulants([C2j_rest_ch2, C2j_task_ch2], 
                        j1=10, j2=14, 
                        title = 'C2(j) EOG channel 2', 
                        labels = [rest_condition, task_condition])



#===============================================================================
# Classification
#===============================================================================
X = np.vstack((H_rest, H_task))
y0 = np.zeros(H_rest.shape[0])
y1 = np.ones(H_task.shape[0])
y = np.hstack((y0,y1))
subject_index = np.hstack((np.arange(n_subjects), np.arange(n_subjects)))


svm = SVC(kernel='linear')
clf = svm

cv  = GroupShuffleSplit(n_splits= 30, 
                              test_size = 0.25, 
                              random_state = 123 )


output = cross_validate(clf, X = X, y = y, scoring = ['accuracy'], cv = cv,
                        groups = subject_index, return_train_score = True,
                        fit_params={}, verbose = 2,
                        n_jobs = 1)

print("Train accuracy = %0.4f +- %0.4f"%(output['train_accuracy'].mean(), output['train_accuracy'].std()))
print("Test accuracy = %0.4f +- %0.4f"%(output['test_accuracy'].mean(), output['test_accuracy'].std()))


plt.show()