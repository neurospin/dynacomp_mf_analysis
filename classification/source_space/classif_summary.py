"""
Plots a summary of the classification results
"""
import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import numpy as np 
import h5py
import os
import classification_utils as clf_utils
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns


from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 20


#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------

def plot_single_learning_curve(train_sizes, test_scores, 
                               title='', label = '', ylim = [0.4, 1.0], 
                               fignum = None):

    if fignum is None:
        plt.figure()
    else:
        plt.figure(fignum)

    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.grid(True)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.25)  #, color="g")

    plt.plot(train_sizes, test_scores_mean, 'o-',label=label)

    
    plt.legend(loc="best")



#======================================================================
# Summary using MF features
#======================================================================

PLOT_LEARNING_CURVES = True
PLOT_POINT           = True

# Chooses the 'representative' point in the learning curve, e.g.,
# if POINT_INDEX = 0, the first point in the learning curve is 
# taken for the pointplot
POINT_INDEX          = -3


#----------------------------------------------------------------------
# Parameters to load data
#----------------------------------------------------------------------
fignum = 'classification  with mf features'

classifiers = ['linear_svm_scaled', 'linear_svm', 'random_forest']
features_names = ['c1', 'c2', 'c1_c2', 'avgC2j', 'maxminC2j']

#~~
features_names_readable = ['$c_1$', '$c_2$', '$c_1$ and $c_2$', 'avg($C_2(j)$)' ,'maxmin($C_2(j)$)']
classifiers_readable    = ['Scaling + Linear SVM', 'Linear SVM', 'Random Forest']
#~~

mf_params_list = [1]
source_rec_params_list = [0]

#~~
formalism_readable = ['p=$\infty$', 'p=1', 'p=2']
source_rec_readable = ['$\lambda = 1/9$', '$\lambda = 1$']
#~~

# Possible conditions for classification
cond0_list = [ ['rest5']     ]
cond1_list = [ ['posttest']    ]

TITLE = '$\mathrm{rest}_5$ versus posttest'
# TITLE = '$\mathrm{rest}_0$ and $\mathrm{rest}_5$ versus pretest and posttest'

conditions_classif = list(zip(cond0_list, cond1_list))


#----------------------------------------------------------------------
# Plot
#----------------------------------------------------------------------

# Data for pointplot
classifier_list    = []
conditions_list    = []
feature_names_list = []
mf_rec_params_list = []
test_score_list    = []

#~~
classifier_list_readable    = []
feature_names_list_readable = []
formalism_list_readable     = []
source_rec_list_readable     = []
#~~

# Plot learning curves in different cases:
for conditions_0, conditions_1 in conditions_classif:
    conditions_folder = '-'.join(conditions_0) + '_' + '-'.join(conditions_1)

    for mf_idx, mf_params in enumerate(mf_params_list):
        for rec_idx, rec_params in enumerate(source_rec_params_list):
            for classif_idx, classif in enumerate(classifiers):
                for feat_idx, feat_name in enumerate(features_names):

                    # File containing classification results
                    filename = '%s_%s_mf_%d_rec_%d.h5'%(classif, feat_name, mf_params, rec_params)
                    outdir   = os.path.join('learning_curves_raw_data', conditions_folder)
                    filename = os.path.join(outdir, filename)

                    if classif == 'random_forest' and (not os.path.isfile(filename)):
                        classif = 'random_forest_no_cv'
                        filename = '%s_%s_mf_%d_rec_%d.h5'%(classif, feat_name, mf_params, rec_params)
                        outdir   = os.path.join('learning_curves_raw_data', conditions_folder)
                        filename = os.path.join(outdir, filename)


                    print(filename)



                    # Load learning curves
                    with h5py.File(filename, "r") as f:
                        # train_sizes is divided by len(conditions_0) so that
                        # train_sizes = number of subjects in training set
                        train_sizes  = f['train_sizes'][:] #/ len(conditions_0)  
                        train_scores = f['train_scores'][:]
                        test_scores  = f['test_scores'][:]

                    # Store data for pointplot
                    test_scores_1d = test_scores[POINT_INDEX,:]
                    temp = 'mf_%d_rec_%d'%(mf_params, rec_params)

                    classifier_list    += [classif]*len(test_scores_1d)
                    conditions_list    += [conditions_folder]*len(test_scores_1d)
                    feature_names_list += [feat_name]*len(test_scores_1d)
                    mf_rec_params_list += [temp]*len(test_scores_1d)
                    test_score_list    += test_scores_1d.tolist()

                    #~~
                    classifier_list_readable    += [classifiers_readable[classif_idx]]*len(test_scores_1d)
                    feature_names_list_readable += [features_names_readable[feat_idx]]*len(test_scores_1d)
                    formalism_list_readable     += [formalism_readable[mf_idx]]*len(test_scores_1d)
                    source_rec_list_readable    += [source_rec_readable[rec_idx]]*len(test_scores_1d)
                    #~~


                    if PLOT_LEARNING_CURVES:
                        #label = '%s, %s_%s_mf_%d_rec_%d'%(classif, conditions_folder, feat_name,mf_params, rec_params)
                        label = formalism_readable[mf_idx]
                        plot_single_learning_curve(train_sizes, test_scores, 
                                                   title=classifiers_readable[classif_idx] + ' - '+ TITLE, 
                                                   label = label,
                                                   ylim = [0.4, 1.0], 
                                                   fignum = fignum)



# Build dataset
data_dict = {'classifier':classifier_list, 
             'conditions':conditions_list,
             'features'  :feature_names_list,
             'mf_rec'    : mf_rec_params_list,
             'Accuracy'     :test_score_list,
             'Classifier':classifier_list_readable,
             'Features'  :feature_names_list_readable,
             'Formalism' : formalism_list_readable,
             'Regularization' : source_rec_list_readable}

df  = pd.DataFrame.from_dict(data_dict)



HUE = df['Features'] # df['conditions'] + ', ' +df['features']+'_'+df['mf_rec']


#--------------------------------------------------------------------------------
plt.figure()
ax = sns.pointplot(x='Accuracy', y='Classifier', data=df,
                   hue = HUE ,
                   capsize=.1,
                   linestyles='', ci="sd")
plt.xlim([0.4, 1.0])
plt.grid()
plt.title(TITLE)
# sns.set(style="darkgrid")
plt.tight_layout()
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
plt.figure()
ax = sns.pointplot(x='Accuracy', y='Regularization', data=df,
                   hue = df['Formalism'] ,  #df['conditions'] + ', ' +df['features']+', '+df['classifier']
                   capsize=.1,
                   linestyles='', ci="sd")
plt.xlim([0.4, 1.0])
plt.title(TITLE)
# sns.set(style="darkgrid")
plt.tight_layout()

plt.grid()
#--------------------------------------------------------------------------------



plt.show()

