"""
Perfoms hypothesis testing on H (self-similarity) and M (multifractality)
and plot on the cortex surface the significant values 

Test for H:
    H0:  H = 0.5
    H1:  H != 0.5
"""

import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import numpy as np
from scipy.stats import ttest_1samp
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
 
# Select condition ('rest0', 'rest5', 'pretest', 'posttest')
condition = 'rest5'

# remove outliers when computing the mean cp for the stc file
outcoef = 2. 

# Hypothesis testing parameters
alpha = 0.05
null_hyp_ttest_1samp = [0.5, 0.0] # null hypothesis for H and M
cp_tail  = [0, 1] # tail for [c1,c2] in the 1-sample t-test
correction_multiple_tests = 'fdr' # 'fdr', 'bonferroni' or None


#===============================================================================
# Functions
#===============================================================================

def compute_pvalue_t_test_1_sample(samples, h0_val = 0, tail = 0):
    """
    Apply 1 sample t test.
    Args:
        h0_val: value of the mean under the null hypothesis
        tail:   0 for 2-tail, 1 for 1-tail
    """
    # remove outliers
    # outliers = (samples - samples.mean()) > 2*samples.std()
    # samples = samples[~outliers]

    result = ttest_1samp(samples, h0_val)
    pval   = result.pvalue
    stat   = result.statistic

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
all_log_cumulants,_ ,subjects_list = \
    mfr.load_data_groups_subjects(condition, groups, subjects)

all_log_cumulants = all_log_cumulants[:, :, :2]  # c3 is not used
n_subjects = all_log_cumulants.shape[0]
n_labels   = all_log_cumulants.shape[1]

c1_array   = all_log_cumulants[:, :, 0]
c2_array   = all_log_cumulants[:, :, 1]


#-------------------------------------------------------------------------------
# Compute p-values
#-------------------------------------------------------------------------------
p_vals = np.ones( (n_labels, 2) ) # shape (n_labels, n_cumulants)

for label in range(n_labels):
    for cumul_idx, cumul in enumerate([1, 2]):

        signal = 1.0
        # Invert signal of c2 to obtain M
        if cumul == 2:
            signal = -1.0

        samples = signal*all_log_cumulants[:, label, cumul_idx]

        # Compute and store p-values
        pval = compute_pvalue_t_test_1_sample(samples,
                                              null_hyp_ttest_1samp[cumul_idx],
                                              cp_tail[cumul_idx])

        p_vals[label, cumul_idx] = pval


#-------------------------------------------------------------------------------
# Apply correction for multiple tests
#-------------------------------------------------------------------------------
if correction_multiple_tests is not None:
    for cumul_idx, cumul in enumerate([1, 2]):
        # get list of p-values for all labels
        pvalues = p_vals[:, cumul_idx].copy()
        # correction
        if correction_multiple_tests == 'fdr':
            # - Benjamini/Hochberg  (non-negative) =  'indep' in mne.fdr_correction
            p_vals[:, cumul_idx] = \
                multipletests(pvalues, alpha, method = 'fdr_bh')[1]

        elif correction_multiple_tests == 'bonferroni':
            # bonferroni
            p_vals[:, cumul_idx] = \
                multipletests(pvalues, alpha, method = 'bonferroni')[1]



#-------------------------------------------------------------------------------
# Organize data to plot
#-------------------------------------------------------------------------------

# means across subjects, removing outliers
c1_mean = my_mean(all_log_cumulants[:, :, 0], axis = 0, outcoef = outcoef)
c2_mean = my_mean(all_log_cumulants[:, :, 1], axis = 0, outcoef = outcoef)


# set values that do not passed the test as the value in the null hypothesis
c1_mean[p_vals[:, 0] >= alpha] =  null_hyp_ttest_1samp[0]
c2_mean[p_vals[:, 1] >= alpha] =  null_hyp_ttest_1samp[1]


#-------------------------------------------------------------------------------
# Plot and save
#-------------------------------------------------------------------------------
c1_filename = os.path.join('output_images', 'c1_' + condition + '.png')
c2_filename = os.path.join('output_images', 'c2_' + condition + '.png')


plots.plot_brain(c1_mean, fmin = 0.8, fmax = 1.20, 
                    png_filename = c1_filename, positive_only = True)
plots.plot_brain(c2_mean, fmin = 0.00, fmax = 0.03, 
                    png_filename = c2_filename, positive_only = True)