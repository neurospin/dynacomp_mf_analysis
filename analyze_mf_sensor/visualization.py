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
sensor_type = 'mag'


# select analysis type: 0 for H, 1 for M
see_cumulant = 1

# Select groups and subjects
groups = ['AV', 'V' ,'AVr']
subjects = info['subjects']

# Select conditions
rest_condition = 'rest5'
task_condition = 'posttest'

# Select MF parameters and source reconstruction parameters
mf_params_idx = 0

# Load cumulants and log-cumulants
all_log_cumulants_rest, all_cumulants_rest, subjects_list, params, _, _, _ = \
     mfr.load_data_groups_subjects(rest_condition, mf_param_idx = mf_params_idx, channel_type = sensor_type)
all_log_cumulants_task, all_cumulants_task, subjects_list, params, channels_picks, channels_names, ch_name2index = \
    mfr.load_data_groups_subjects(task_condition, mf_param_idx = mf_params_idx, channel_type = sensor_type)



#===============================================================================
# Averages and topomaps
#===============================================================================

# Load raw to get info about sensor positions
raw = mne.io.read_raw_fif(raw_filename)

# get sensor positions via layout
pos = mne.find_layout(raw.info).pos[channels_picks, :]

# compute averages across subjects
avg_log_cumulants_rest = all_log_cumulants_rest.mean(axis = 0)
avg_log_cumulants_task = all_log_cumulants_task.mean(axis = 0)

# compute stds across subjects
std_log_cumulants_rest = all_log_cumulants_rest.std(axis = 0)
std_log_cumulants_task = all_log_cumulants_task.std(axis = 0)

# Plot
vmin = np.min(avg_log_cumulants_task[:, 0])
vmax = np.max(avg_log_cumulants_task[:, 0])

if see_cumulant == 0:
    v_utils.plot_data_topo(avg_log_cumulants_rest[:, 0], pos, vmin = vmin, vmax = vmax, title = 'H rest')
    v_utils.plot_data_topo(avg_log_cumulants_task[:, 0], pos, vmin = vmin, vmax = vmax, title = 'H task')
    # plot_data_topo(std_log_cumulants_rest[:, 0], pos, title = 'std H rest')
    # plot_data_topo(std_log_cumulants_task[:, 0], pos, title = 'std H task')


vmin = -np.min(avg_log_cumulants_task[:, 1])
vmax = -np.max(avg_log_cumulants_task[:, 1])

if see_cumulant == 1:
    v_utils.plot_data_topo(-1*avg_log_cumulants_rest[:, 1].clip(max = 0), pos, vmin = vmin, vmax = vmax, title = 'M rest')
    v_utils.plot_data_topo(-1*avg_log_cumulants_task[:, 1].clip(max = 0), pos, vmin = vmin, vmax = vmax, title = 'M task')


if see_cumulant == 0:
    v_utils.plot_data_topo(avg_log_cumulants_rest[:, 0] - avg_log_cumulants_task[:, 0], pos, title = 'H rest - task', cmap = 'Reds')
# plot_data_topo(avg_log_cumulants_task[:, 1].clip(max = 0) - avg_log_cumulants_rest[:, 1].clip(max = 0), pos, title = 'M rest - task')


#===============================================================================
# C1(j) and C2(j)
#===============================================================================

# compute averages across subjects
avg_cumulants_rest = all_cumulants_rest.mean(axis = 0) # shape (n_channels, 3,15)
avg_cumulants_task = all_cumulants_task.mean(axis = 0)


# compare average of C_2(j) over [j1=9, j2=13]
avg_cumulants_rest_9_13 = (avg_cumulants_rest[:,:,8:13]).max(axis = 2)
avg_cumulants_task_9_13 = (avg_cumulants_task[:,:,8:13]).max(axis = 2)

vmin = np.min(avg_cumulants_task_9_13[:, 1])
vmax = np.max(avg_cumulants_task_9_13[:, 1])

if see_cumulant == 1:
    v_utils.plot_data_topo(avg_cumulants_rest_9_13[:, 1], pos, vmin = vmin, vmax = vmax, title = 'Average C_2(j) rest')
    v_utils.plot_data_topo(avg_cumulants_task_9_13[:, 1], pos, vmin = vmin, vmax = vmax, title = 'Average C_2(j) task')
    v_utils.plot_data_topo(avg_cumulants_rest_9_13[:, 1]-avg_cumulants_task_9_13[:, 1], pos, title = 'Average C_2(j) rest-task', cmap = 'Reds')


if sensor_type == 'mag':

    if see_cumulant == 0:
        # Sensor for rest/task comparison
        sensor1_name = 'MEG0341'
        sensor1_index = ch_name2index[sensor1_name]

        v_utils.plot_cumulants( [avg_cumulants_rest[sensor1_index, 0, :], avg_cumulants_task[sensor1_index, 0, :] ],
                        title ='H rest/task - ' + sensor1_name,
                        labels = ['rest', 'task'])

        # Sensors for comparison of different regions (rest)
        sensor2_name = 'MEG0811'
        sensor3_name = 'MEG1841'
        sensor2_index = ch_name2index[sensor2_name]
        sensor3_index = ch_name2index[sensor3_name]

        v_utils.plot_cumulants( [avg_cumulants_rest[sensor2_index, 0, :], avg_cumulants_rest[sensor3_index, 0, :] ],
                        title ='H rest - ' + sensor2_name + ' vs. ' + sensor3_name,
                        labels = [sensor2_name, sensor3_name])


    if see_cumulant == 1:
        # Sensor compare M rest vs task
        sensor4_name = 'MEG2621'
        sensor4_index = ch_name2index[sensor4_name]
        v_utils.plot_cumulants( [avg_cumulants_rest[sensor4_index, 1, :], avg_cumulants_task[sensor4_index, 1, :] ],
                        title ='M rest/task - ' + sensor4_name,
                        labels = ['rest', 'task'])

        sensor5_name = 'MEG1811'
        sensor5_index = ch_name2index[sensor5_name]
        v_utils.plot_cumulants( [avg_cumulants_rest[sensor5_index, 1, :], avg_cumulants_task[sensor5_index, 1, :] ],
                        title ='M rest/task - ' + sensor5_name,
                        labels = ['rest', 'task'])


    if see_cumulant == 0:
        v_utils.plot_sensors([sensor1_name], pos, ch_name2index)
        v_utils.plot_sensors([sensor2_name, sensor3_name], pos, ch_name2index)
    if see_cumulant == 1:
        v_utils.plot_sensors([sensor4_name], pos, ch_name2index)
        v_utils.plot_sensors([sensor5_name], pos, ch_name2index)


#===============================================================================
# C1(j) and C2(j) for all sensors
#===============================================================================
avg_all_sensors_cumulants_rest = avg_cumulants_rest.mean(axis = 0) # shape (3,15)
avg_all_sensors_cumulants_task = avg_cumulants_task.mean(axis = 0)
std_all_sensors_cumulants_rest = avg_cumulants_rest.std(axis = 0)  # shape (3,15)
std_all_sensors_cumulants_task = avg_cumulants_task.std(axis = 0)


if see_cumulant == 0:
    plt.figure()
    plt.title('C_1(j) - average over all sensors')
    plt.errorbar(np.arange(1, 15), avg_all_sensors_cumulants_rest[0, :],fmt ='bo--', label = 'rest', yerr = std_all_sensors_cumulants_rest[0, :])
    plt.errorbar(np.arange(1, 15), avg_all_sensors_cumulants_task[0, :], fmt ='ro--', label = 'task', yerr = std_all_sensors_cumulants_task[0, :])
    plt.xlabel('j')
    plt.grid()
    plt.legend()

if see_cumulant == 1:
    plt.figure()
    plt.title('C_2(j) - average over all sensors')
    plt.errorbar(np.arange(1, 15), avg_all_sensors_cumulants_rest[1, :],fmt ='bo--', label = 'rest', yerr = std_all_sensors_cumulants_rest[1, :])
    plt.errorbar(np.arange(1, 15), avg_all_sensors_cumulants_task[1, :], fmt ='ro--', label = 'task', yerr = std_all_sensors_cumulants_task[1, :])
    plt.xlabel('j')
    plt.grid()
    plt.legend()

raw.plot_sensors()
plt.show()

