import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 20


import source_mf_results as mfr_source
import meg_info as meg_info_source
info_source   = meg_info_source.get_info()

import pandas as pd

#----------------------------------------------------------------------------------------------
# Options
#----------------------------------------------------------------------------------------------
space = 'source'

# mf_params_idx = 2
source_rec_params_idx = 0 


clip_c2 = True # set c_2 to zero if c2 > 0

mfidx2formalism = {0:'WCMF', 1:'WLMF', 2:'p = 1', 3:'p = 2'}
p2mfidx = {np.inf:1, 1:2, 2:3}

p_list = [1, 2, np.inf]

conditions_0 = ['rest5']
conditions_1 = ['posttest']

log_cumulants = [1]



#----------------------------------------------------------------------------------------------
# Run 
#----------------------------------------------------------------------------------------------
# Select groups and subjects
groups = ['AV', 'V', 'AVr']
subjects = {}
subjects['AV']  = info_source['subjects']['AV']
subjects['V']   = info_source['subjects']['V']
subjects['AVr'] = info_source['subjects']['AVr']

mean_diff = np.zeros(len(p_list))
std_diff  = np.zeros(len(p_list))


diff_list_df = []
p_list_idx_df    = [] 
p_list_val_df    = [] 
for ii, p in enumerate(p_list):
    mf_params_idx = p2mfidx[p]

    X, y, subjects_idx, groups_idx, subjects_list =  mfr_source.load_logcumul_for_classification(conditions_0, conditions_1,
                                                                     log_cumulants,
                                                                     groups, subjects,
                                                                     mf_params_idx, source_rec_params_idx,
                                                                     clip_c2 = clip_c2)

    X0 = X[y==0, :].T  # (cortical regions, subjects)
    X1 = X[y==1, :].T  #

    diffX = np.mean(X1 - X0, axis = 0)


    mean_diff[ii] = diffX.mean()
    std_diff[ii]  = diffX.std()

    diff_list_df += diffX.tolist()
    p_list_val_df    += [p]*len(diffX)
    p_list_idx_df    += [ii]*len(diffX)


ystring = 'mean difference in $c_%d$'%(log_cumulants[0]+1)
dataset = {'p':p_list_val_df, 'p_idx': p_list_idx_df ,ystring:diff_list_df}

df  = pd.DataFrame.from_dict(dataset)

plt.figure()
ax = sns.pointplot(x='p_idx', y=ystring, data=df,
                   hue = df['p'],
                   capsize=.1,
                   linestyles='',
                   ci="sd")

if log_cumulants[0]==0:
    plt.ylim([-0.175, 0.015])

if log_cumulants[0]==1:
    plt.ylim([-0.02, 0.015])

plt.title( ' '.join(conditions_0) + ' versus ' +  ' '.join(conditions_1))

plt.grid()

plt.xlabel('')
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])


plt.show()