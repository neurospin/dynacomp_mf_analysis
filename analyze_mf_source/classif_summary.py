"""
Plots a summary of the classification results
"""
import numpy as np 
import h5py
import os
import classification_utils as clf_utils
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

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

classifiers = ['linear_svm']
features_names = ['c1_c2', 'c1']

mf_params_list = [1]
source_rec_params_list = [0]

# Possible conditions for classification
cond0_list = [ ['rest5'],    ['rest0', 'rest5'],       ]
cond1_list = [ ['posttest'], ['pretest', 'posttest']   ]

conditions_classif = list(zip(cond0_list, cond1_list))


#----------------------------------------------------------------------
# Plot
#----------------------------------------------------------------------

# Data for pointplot
classifier_list    = []
conditions_list    = []
feature_names_list = []
test_score_list    = []

# Plot learning curves in different cases:
for conditions_0, conditions_1 in conditions_classif:
    conditions_folder = '-'.join(conditions_0) + '_' + '-'.join(conditions_1)

    for mf_params in mf_params_list:
        for rec_params in source_rec_params_list:
            for classif in classifiers:
                for feat_name in features_names:
                    # File containing classification results
                    filename = '%s_%s_mf_%d_rec_%d.h5'%(classif, feat_name, mf_params, rec_params)
                    outdir   = os.path.join('learning_curves_raw_data', conditions_folder)
                    filename = os.path.join(outdir, filename)
                    print(filename)

                    # Load learning curves
                    with h5py.File(filename, "r") as f:
                        # train_sizes is divided by len(conditions_0) so that
                        # train_sizes = number of subjects in training set
                        train_sizes  = f['train_sizes'][:] / len(conditions_0)  
                        train_scores = f['train_scores'][:]
                        test_scores  = f['test_scores'][:]

                    # Store data for pointplot
                    test_scores_1d = test_scores[POINT_INDEX,:]
                    temp = '%s_mf_%d_rec_%d'%(feat_name,mf_params, rec_params)

                    classifier_list    += [classif]*len(test_scores_1d)
                    conditions_list    += [conditions_folder]*len(test_scores_1d)
                    feature_names_list += [temp]*len(test_scores_1d)
                    test_score_list    += test_scores_1d.tolist()


                    if PLOT_LEARNING_CURVES:
                        label = '%s, %s_%s_mf_%d_rec_%d'%(classif, conditions_folder, feat_name,mf_params, rec_params)
                        plot_single_learning_curve(train_sizes, test_scores, 
                                                   title='', 
                                                   label = label,
                                                   ylim = [0.4, 1.0], 
                                                   fignum = fignum)

# Build dataset
data_dict = {'classifier':classifier_list, 
             'conditions':conditions_list,
             'features'  :feature_names_list,
             'score'     :test_score_list}
df  = pd.DataFrame.from_dict(data_dict)

plt.figure()
ax = sns.pointplot(x='score', y='classifier', data=df,
                   hue = df['conditions'] + ', ' +df['features'],
                   capsize=.1)
plt.xlim([0.4, 1.0])
plt.grid()
# sns.set(style="darkgrid")

plt.show()

