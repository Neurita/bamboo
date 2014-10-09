"""Features: Utilities related with pandas DataFrame features.
"""
# Author: Borja Ayerdi <ayerdi.borja@gmail.com>
# License: BSD 3 clause
# Copyright: UPV/EHU

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_validation import LeaveOneOut
from sklearn.ensemble import ExtraTreesClassifier

def get_feature_importance(X,Y):
    """
    Given data matrix (X) and labels (Y) we get feature importance using
    Leave One Out and ExtraTrees classifier.

    Parameters
    ----------
    X: pd.DataFrame
        DataFrame with data (n_samples, n_features)
    Y: pd.DataFrame
        DataFrame with labels (n_samples, 1)

    Returns
    -------
    pd.DataFrame
        DataFrame with feature importance values.

    """
    # Leave One Out
    K = len(Y)
    vAcc = []
    loo = LeaveOneOut(n=K)
    yy = np.zeros(len(Y))
    feat_imp = np.zeros((1,X.shape[1]))
        
    for train, test in loo:
        x_train, x_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]

        # We correct NaN values in x_train and x_test
        nan_mean = stats.nanmean(x_train)
        nan_train = np.isnan(x_train)
        nan_test = np.isnan(x_test)

        x_test[nan_test] = 0
        x_test = x_test + nan_test*nan_mean

        x_train[nan_train] = 0
        x_train = x_train + nan_train*nan_mean

        # Compute mean, std and noise for z-score
        std = np.std(x_train,axis=0)
        med = np.mean(x_train,axis=0)
        noise = [np.random.uniform(-0.000005, 0.000005) for p in range(0,x_train.shape[1])]

        # Apply Z-score
        x_train = (x_train-med)/(std+noise)
        x_test = (x_test-med)/(std+noise)

        # Classifier type.
        classifier = ExtraTreesClassifier()
        classifier = classifier.fit(x_train, y_train)
        
        feat_imp_np = np.array(classifier.feature_importances_)
        
        feat_imp = feat_imp + feat_imp_np
    
    res = np.around(feat_imp/x_train.shape[0], decimals=4)

    return pd.DataFrame(res)  
