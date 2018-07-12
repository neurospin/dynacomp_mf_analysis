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


#===============================================================================
# Global parameters 
#===============================================================================
# Load info
import meg_info
info   = meg_info.get_info()

# Select groups and subjects
groups = ['AV']
subjects = {}
subjects['AV'] = info['subjects']['AV']

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
            _, all_cumulants ,subjects_list = \
                mfr.load_data_groups_subjects(condition, groups, subjects)

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

                ax.grid()
                ax.set_xlabel('scale j')
                ax.set_ylabel('C%d(j)'%(cumul_idx+1))
                ax.legend()
                ax.set_title(condition)
                color_idx += 1

            
    plt.show()

if __name__ == '__main__':
    run()