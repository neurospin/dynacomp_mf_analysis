"""
Performs the following tests:

- For H:
	H0:  Hrest = Htask
	H1:  Hrest > Htask

- For M:
	H0:  Mrest = Mtask
	H1:  Mrest < Mtask

!!! Important:

positive values of c2 are set to 0 before the test
"""

import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import source_mf_results as mfr
import plots


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
 
# Select conditions to contrast ('rest0', 'rest5', 'pretest', 'posttest')
conditions = ['rest5', 'posttest']  # contrast image: conditions[0] - conditions[1]
test_variable = 1 # select 0 for H or 1 for M


# remove outliers when computing the mean cp for the stc file
outcoef = 2. 

# Hypothesis testing parameters
alpha = 0.05
cp_tail  = [1, -1] # tail for [c1,c2] in the 1-sample t-test
correction_multiple_tests = 'fdr' # 'fdr', 'bonferroni' or None




#===============================================================================
# Functions
#===============================================================================

def compute_pvalue_t_test_rel(x, y, tail = 0):
    """
    Apply t test for 2 related samples x and y
    Args:
        tail:   0 for 2-tail, +1 or -1 for 1-tail
    """
    # remove outliers
    # outliers = (samples - samples.mean()) > 2*samples.std()
    # samples = samples[~outliers]

    result = ttest_rel(x, y)
    pval   = result.pvalue
    stat   = result.statistic

    # positive statistic: x > y
    # tail = 1 -> test H1: x>y

    if tail!=0:
        pval = 0.5*(1+np.sign(tail)*np.sign(stat)*(pval-1))
    return pval


def my_mean(a,axis=0,outcoef=None):
    x= a.copy()
    #print (x)
    #remove NaN values
    mask= ~np.isnan(x)
    x[~mask]=0
    y= np.sum(x,axis)/np.sum(mask,axis)
    if outcoef:
        x[~mask]=np.nan
        mask2= np.abs(y-x) > outcoef*my_std(x,axis)
        x[mask2]=np.nan
        y = my_mean(x,axis)
    return y

def my_std(a,axis=0):
    x= a.copy()
    #remove NaN values
    mask= ~np.isnan(x)
    x[~mask]=0
    y= np.sqrt(np.sum((x-np.sum(x,axis)/np.sum(mask,axis))**2,axis)/np.sum(mask,axis))
    return y


#===============================================================================
# Run
#===============================================================================

# load log-cumulants
# array all_log_cumulants: shape (n_subjects, n_labels, n_cumul)
all_log_cumulants_cond_0,_ ,subjects_list = \
    mfr.load_data_groups_subjects(conditions[0], groups, subjects)

all_log_cumulants_cond_1,_ ,subjects_list = \
    mfr.load_data_groups_subjects(conditions[1], groups, subjects)


n_subjects = all_log_cumulants_cond_0.shape[0]
n_labels   = all_log_cumulants_cond_0.shape[1]

c1_array_cond_0   = all_log_cumulants_cond_0[:, :, 0]
c1_array_cond_1   = all_log_cumulants_cond_1[:, :, 0]
c2_array_cond_0   = all_log_cumulants_cond_0[:, :, 1].clip(max = 0)
c2_array_cond_1   = all_log_cumulants_cond_1[:, :, 1].clip(max = 0)


#-------------------------------------------------------------------------------
# Compute p-values
#-------------------------------------------------------------------------------
p_vals = np.ones( n_labels ) # shape (n_labels,)

for label in range(n_labels):

	if test_variable == 0:
		samples_0 = c1_array_cond_0[:, label]
		samples_1 = c1_array_cond_1[:, label]

	if test_variable == 1:
		samples_0 = -1*c2_array_cond_0[:, label] # invert signal to get M
		samples_1 = -1*c2_array_cond_1[:, label]

	pval = compute_pvalue_t_test_rel(samples_0, samples_1, cp_tail[test_variable])
	p_vals[label] = pval
	# # Compute and store p-values
 #    pval = compute_pvalue_t_test_rel(samples_0, samples_1, cp_tail[test_variable])
 #    p_vals[label] = pval


#-------------------------------------------------------------------------------
# Apply correction for multiple tests
#-------------------------------------------------------------------------------
if correction_multiple_tests is not None:
    # correction
    if correction_multiple_tests == 'fdr':
        # - Benjamini/Hochberg  (non-negative) =  'indep' in mne.fdr_correction
        p_vals = \
            multipletests(p_vals, alpha, method = 'fdr_bh')[1]

    elif correction_multiple_tests == 'bonferroni':
        # bonferroni
        p_vals = \
            multipletests(p_vals, alpha, method = 'bonferroni')[1]

#-------------------------------------------------------------------------------
# Organize data to plot
#-------------------------------------------------------------------------------

# contrast across subjects, removing outliers
if test_variable == 0:
	contrast = my_mean(c1_array_cond_0, axis = 0, outcoef = outcoef) \
			   - my_mean(c1_array_cond_1, axis = 0, outcoef = outcoef)
if test_variable == 1:
	contrast = my_mean(-c2_array_cond_0, axis = 0, outcoef = outcoef) \
			   - my_mean(-c2_array_cond_1, axis = 0, outcoef = outcoef)


# set values that do not passed the test as the value in the null hypothesis
contrast[p_vals >= alpha] =  0


#-------------------------------------------------------------------------------
# Plot and save
#-------------------------------------------------------------------------------
if test_variable == 0:
	cumulant_name = 'c1'
if test_variable == 1:
	cumulant_name = 'c2'


filename = '%s_contrast_%s_%s.png'%(cumulant_name, conditions[0], conditions[1])
filename = os.path.join('output_images', filename)

maxval = np.abs(contrast).max()
plots.plot_brain(contrast, fmin = -maxval, fmax = maxval, 
                 png_filename = filename, positive_only = False)
