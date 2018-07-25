"""
For a single subject/condition, plot multifractal spectrum of
EOG channels.
"""

import os, sys
from os.path import dirname, abspath
# add project dir to path
project_dir = dirname(dirname(abspath(__file__))) 
sys.path.insert(0, project_dir)

import numpy as np 
import mfanalysis as mf
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os.path as op
import mne
import os
import h5py

import preprocessing 

import meg_info_sensor_space as meginfo
import mf_config_sensor_space
info =  meginfo.get_info()

import meg_info as meginfo_rec
info_rec   = meginfo_rec.get_info()


SCALE_1 = 10
SCALE_2 = 14   # fs = 2000, f1 = 0.1, f2 = 1.5


SENSOR_NAME = 'MEG0811'

def get_scales(fs, min_f, max_f):
    """
    Compute scales corresponding to the analyzed frequencies
    """
    f0 = (3.0/4.0)*fs
    j1 = int(np.ceil(np.log2(f0/max_f)))
    j2 = int(np.ceil(np.log2(f0/min_f)))
    return j1, j2



# MF parameters
mf_params = mf_config_sensor_space.get_mf_params()
params_index = 1

# Subjects and conditions
groups   = ['AV', 'V', 'AVr']
subjects = {}
subjects['AV'] = info['subjects']['AV']
subjects['V'] = info['subjects']['V']
subjects['AVr'] = info['subjects']['AVr']
conditions  = info['sessions']


# Select subject and condition
group   = 'AV'
subject = subjects[group][0]
condition = 'posttest'
bis     = False

#---------------------------------------------------------
# Load EOG data
#---------------------------------------------------------

# Get raw data
raw = preprocessing.get_raw(group, subject, condition, bis)

# Pick MEG magnetometers or gradiometers
picks =  mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, eog=False,
                        exclude='bads')

picks_ch_names = [raw.ch_names[i] for i in picks]


name2picks = dict(zip(picks_ch_names, picks))
sensor_idx = name2picks[SENSOR_NAME]



data = raw.get_data(picks)
data_sensor = data[sensor_idx, :]



#---------------------------------------------------------
# Apply ICA
#---------------------------------------------------------
if 'rest' in condition:
    icagroup='rest'
elif np.array(['test' in condition,'learn' in condition]).any():
    icagroup='task'

ica_f_name = os.path.join(info_rec['paths_to_subjects'][group][subject],
                                group +'_'+icagroup+'-nofilt-all-ica.fif')



ica = mne.preprocessing.read_ica(ica_f_name).copy()
raw.load_data()
raw = ica.apply(raw)


data_ica = raw.get_data(picks)
data_sensor_ica = data_ica[sensor_idx, :]

#---------------------------------------------------------
# Run MF analysis
#---------------------------------------------------------
params = mf_params[params_index]
mfa = mf.MFA(**params)
mfa.j1 = SCALE_1
mfa.j2 = SCALE_2
mfa.q = np.arange(-8,9)
mfa.verbose = 1

mfa.analyze(data_sensor)
mfa.cumulants.plot(fignum='before ICA')

mfa.analyze(data_sensor_ica)
mfa.cumulants.plot(fignum='after ICA')

plt.show()