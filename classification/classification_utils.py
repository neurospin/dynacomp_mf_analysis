import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, cross_validate, learning_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#===============================================================================
# Definition of classifiers/estimators
#===============================================================================
def get_classifier(classifier_name, inner_cv, groups_cv):
    clf = None
    fit_params = {}
    if classifier_name == 'linear_svm':
        svm = SVC(kernel='linear')
        # parameters for grid search
        p_grid = {}
        p_grid['C'] = np.power(10.0, np.linspace(-4, 4, 10))
        # classifier
        clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
        # parameters required to fit the classifier
        fit_params = {'groups':groups_cv}

    elif classifier_name == 'linear_svm_scaled':
        svm = SVC(kernel='linear')
        # parameters for grid search
        p_grid = {}
        p_grid['C'] = np.power(10.0, np.linspace(-4, 4, 10))
        # classifier
        clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
        # parameters required to fit the classifier
        fit_params = {'gridsearchcv__groups':groups_cv}
        clf = make_pipeline(StandardScaler(),clf)

    elif classifier_name == 'logregl1':
        logregl1 = LogisticRegression(penalty = 'l1')
        # parameters for grid search
        p_grid = {}
        p_grid['C'] = np.power(10.0, np.linspace(-4, 4, 10))       
        # classifier
        clf = GridSearchCV(estimator=logregl1, param_grid=p_grid, cv=inner_cv)
        # parameters required to fit the classifier
        fit_params = {'groups':groups_cv}

    elif classifier_name == 'logregl1_scaled':
        logregl1 = LogisticRegression(penalty = 'l1')
        # parameters for grid search
        p_grid = {}
        p_grid['C'] = np.power(10.0, np.linspace(-4, 4, 10))       
        # classifier
        clf = GridSearchCV(estimator=logregl1, param_grid=p_grid, cv=inner_cv)
        # parameters required to fit the classifier
        fit_params = {'gridsearchcv__groups':groups_cv}
        clf = make_pipeline(StandardScaler(),clf)

    elif classifier_name == 'random_forest':
        rf = RandomForestClassifier(n_estimators=150)
        p_grid = {'max_features': ['sqrt'],
                       'criterion': ['gini', 'entropy'],
                       'max_depth': [2, 8]}

        clf = GridSearchCV(estimator=rf, param_grid=p_grid, cv=inner_cv)
        # parameters required to fit the classifier
        fit_params = {'groups':groups_cv}

    return clf, fit_params


def get_feature_importances(clf, fit_params, X, y):
    w = None
    positive_only = None

    clf.fit(X, y, **fit_params)
    try:
        w = clf.best_estimator_.coef_.squeeze()
        positive_only = False
        return w, positive_only
    except:
        pass

    try:
        w = clf.best_estimator_.feature_importances_
        positive_only = True
        return w, positive_only
    except:
        pass

    try:
        w = clf.steps[1][1].best_estimator_.coef_.squeeze()
        positive_only = False
        return w, positive_only
    except:
        pass


def my_learning_curve(classifier_name, X, y, groups, train_sizes, scoring, 
                     n_splits, random_state, n_jobs):
    """
    train_sizes: array of floats between 0 and 1
    n_splits:    number of cross validation splits
    """
    print("*** Computing learning curve ***")

    train_sizes_abs = []
    train_scores    = []
    test_scores     = []

    for train_s in train_sizes:
        print("---- Running for train size = ", train_s)
        test_size = 1.0 - train_s  

        inner_cv  = GroupShuffleSplit(n_splits= n_splits, 
                                      test_size = test_size, 
                                      random_state = random_state )
        outer_cv  = GroupShuffleSplit(n_splits= n_splits, 
                                      test_size = test_size, 
                                      random_state = random_state )


        # For testing
        for train_index, test_index in outer_cv.split(X, y, groups = groups):
            #print("TRAIN:", train_index, "TEST:", test_index)
            print(groups[train_index])
            print(groups[test_index])
            print("TRAIN:", len(train_index), "TEST:", len(test_index))
            print("     ")


        clf, fit_params = get_classifier(classifier_name, inner_cv, groups)

        output = cross_validate(clf, X = X, y = y, scoring = scoring, cv = outer_cv,
                        groups = groups, return_train_score = True,
                        fit_params=fit_params, verbose = 2,
                        n_jobs = n_jobs)

        print("Train accuracy = %0.4f +- %0.4f"%(output['train_accuracy'].mean(), output['train_accuracy'].std()))
        print("Test accuracy = %0.4f +- %0.4f"%(output['test_accuracy'].mean(), output['test_accuracy'].std()))


        n_samples = X.shape[0]
        train_sizes_abs.append( np.ceil(n_samples*train_s)  )
        train_scores.append(output['train_accuracy'])
        test_scores.append(output['test_accuracy'])


    return np.array(train_sizes_abs), np.array(train_scores), np.array(test_scores)


def plot_learning_curve(train_sizes, train_scores, test_scores, 
                        title='', ylim = [0.4, 1.0], 
                        fignum = None, show_train_curves = True):

    if fignum is None:
        plt.figure()
    else:
        plt.figure(fignum)

    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    if show_train_curves:
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    if show_train_curves:
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")


def run_classification(classifier_name, 
                       X, y, groups_cv,
                       train_sizes,
                       scoring,
                       n_splits,
                       random_state,
                       n_jobs, 
                       ref_train_size = 0.8):

    # Get learning curve
    train_sizes_abs, train_scores, test_scores = \
                            my_learning_curve(classifier_name, 
                                              X, y, 
                                              groups_cv, 
                                              train_sizes, 
                                              scoring, 
                                              n_splits, 
                                              random_state,
                                              n_jobs)

    # Get weights/feature importances
    inner_cv  = GroupShuffleSplit(n_splits= n_splits, 
                                 test_size = 1-ref_train_size, 
                                 random_state = random_state)

    clf, fit_params = get_classifier(classifier_name, inner_cv, groups_cv)
    w, positive_only = get_feature_importances(clf, fit_params, X, y)


    return train_sizes_abs, train_scores, test_scores, w, positive_only