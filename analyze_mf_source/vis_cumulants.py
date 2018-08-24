"""
Visualization of the cumulants C_1(j) and C_2(j), as in figures 4
and 5 of the paper.
"""

import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import numpy as np
import source_mf_results as mfr
import matplotlib.pyplot as plt
from scipy.stats import linregress

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 16



OPTION = 0

if OPTION == 0:
    EXTRA_INFO =  ''
    TIMESTR    =  '20180713'
else:
    EXTRA_INFO = 'no_ica_'  # ''
    TIMESTR    = '20180724' # '20180713'

#===============================================================================
# Global parameters 
#===============================================================================
# Load info
import meg_info
info   = meg_info.get_info()

# Select groups and subjects
groups = ['AV', 'V', 'AVr']
subjects = {}
subjects['AV'] = info['subjects']['AV']
subjects['V'] = info['subjects']['V']
subjects['AVr'] = info['subjects']['AVr']

# Select MF parameters and source reconstruction parameters
mf_params_idx = 1
source_rec_params_idx = 0
 
# Select conditions ('rest0', 'rest5', 'pretest', 'posttest')
# and cortical labels
conditions = ['rest5', 'posttest']
labels     = [110, 20]
labels_region = ['frontal', 'occipital']
colors     = ['r', 'b']

def run():
    log2e = np.log2(np.exp(1))

    for cumul_idx in [0, 1]:
        fig, axes = plt.subplots(1,2)    
        for cond_idx, condition in enumerate(conditions):
            # load cumulants
            # array all_cumulants: shape (n_subjects, n_labels, n_cumul, max_j)
            all_log_cumulants, all_cumulants ,subjects_list = \
                mfr.load_data_groups_subjects(condition, groups, subjects,
                                              mf_param_idx = mf_params_idx, 
                                              source_rec_param_idx = source_rec_params_idx,
                                              time_str = TIMESTR,
                                              extra_info = EXTRA_INFO)

            color_idx = 0
            for label, region in zip(labels, labels_region):

                Cj = all_cumulants[:, label, cumul_idx, :].mean(axis = 0)
                max_j = len(Cj)

                ax = axes[cond_idx]

                ax.plot(np.arange(1, max_j+1), Cj, colors[color_idx]+'o--',
                        alpha = 0.5)

                # linear regression
                x = np.arange(8, 13) # scales 8 to 12
                y = Cj[7:12]

                slope, intercept, r_value, p_value, std_err = linregress(x,y)

                x0 = 8
                x1 = 12
                y0 = slope*x0 + intercept
                y1 = slope*x1 + intercept

                ax.plot([x0, x1], [y0, y1], colors[color_idx]+'-', linewidth=2,
                        label = region + ', slope*log2(e) = %0.3f'%(log2e*slope))
                #

                ax.grid(True)
                ax.set_xlabel('scale j')
                ax.set_ylabel('$C_%d(j)$'%(cumul_idx+1))
                ax.legend()
                ax.set_title(condition)
                color_idx += 1



    all_log_cumulants_rest, _, _ = \
                mfr.load_data_groups_subjects('rest5', groups, subjects,
                                              mf_param_idx = mf_params_idx, 
                                              source_rec_param_idx = source_rec_params_idx,
                                              time_str = TIMESTR,
                                              extra_info = EXTRA_INFO)

    all_log_cumulants_task, _, _  = \
                mfr.load_data_groups_subjects('posttest', groups, subjects,
                                              mf_param_idx = mf_params_idx, 
                                              source_rec_param_idx = source_rec_params_idx,
                                              time_str = TIMESTR,
                                              extra_info = EXTRA_INFO)


    c2_rest_occ = all_log_cumulants_rest[:, 20, 1]
    c2_task_occ = all_log_cumulants_task[:, 20, 1]
            
    return c2_rest_occ, c2_task_occ

    

if __name__ == '__main__':
    c2_rest_occ, c2_task_occ = run()

    plt.show()