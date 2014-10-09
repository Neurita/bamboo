"""Classification: Utilities for data classification using pandas DataFrame.
"""
# Author: Borja Ayerdi <ayerdi.borja@gmail.com>
# License: BSD 3 clause
# Copyright: UPV/EHU

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn.svm import SVC
from sklearn.mixture import GMM
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier


def make_classification(X, Y, classif=1):
    """
    Give data(X) and labels(Y) perform Leave One Out classification with selected classifier.

    Parameters
    ----------
    X: pd.DataFrame
        DataFrame with data (n_samples, n_features)
    Y: pd.DataFrame
        DataFrame with labels (n_samples, 1)
    classif: int, optional (default=1)
        # 1. Decision Tree
        # 2. Random Forest
        # 3. SVM (RBF)
        # 4. SVM (Linear)
        # 5. 1-NN
        # 6. 3-NN
        # 7. 5-NN
        # 8. 10-NN
        # 9. Naive Bayes
        # 10. SGD

    Returns
    -------
    list
        Accuracy, Sensitivity, Specificity and ROC values

    """

    # Leave One Out
    K = len(Y)
    vAcc = []
    loo = LeaveOneOut(n=K)
    yy = np.zeros(len(Y))
  
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
        if classif == 1:
            classifier = DecisionTreeClassifier()
        elif classif == 2:
            classifier = RandomForestClassifier()
        elif classif == 3:
            classifier = SVC()
        elif classif == 4:    
            classifier = LinearSVC()
        elif classif == 5:
            classifier = KNeighborsClassifier(n_neighbors=1)
        elif classif == 6:
            classifier = KNeighborsClassifier(n_neighbors=3)
        elif classif == 7:
            classifier = KNeighborsClassifier(n_neighbors=5)
        elif classif == 8:
            classifier = KNeighborsClassifier(n_neighbors=10)
        elif classif == 9:
            classifier = GaussianNB()
        elif classif == 10:
            classifier = SGDClassifier()
        else:
            classifier = SGDClassifier()
        
        classifier = classifier.fit(x_train, y_train)

        # For testing x_test & y_test
        y_pred = classifier.predict(x_test)  
        yy[test[0]] = y_pred[0]
            
        cm = confusion_matrix(y_test, y_pred)
        acc = float(cm.trace())/cm.sum()
        vAcc.append(acc)
        
    cmm = confusion_matrix(yy, Y)

    bd_std = np.std(vAcc)
    bd_acc = np.mean(vAcc)
    sens = cmm[0,0]/float(cmm[:,0].sum())
    spec = cmm[1,1]/float(cmm[:,1].sum())
    
    #roc_auc = 0
    fpr, tpr, thresholds = roc_curve(yy,Y)
    roc_auc = auc(fpr, tpr)
    
    # We return Accuracy, Sensitivity, Specificity and ROC
    return format(bd_acc*100,'.2f'),format(sens*100,'.2f'),format(spec*100,'.2f'),format(roc_auc,'.2f')


def param_experiments(df, control_index, patient_index, classifier_type=1):
    """
    Function to perform experiments easily, given a DataFrame, control and patient index and classifier type.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with data (n_samples, n_features)
    control_index: pd.DataFrame
        Control index DataFrame containing True/False values. 
    patient_index: pd.DataFrame
        Patient index DataFrame containing True/False values. 

    classifier_type: int, optional (default=1)
        # 1. Decision Tree
        # 2. Random Forest
        # 3. SVM (RBF)
        # 4. SVM (Linear)
        # 5. 1-NN
        # 6. 3-NN
        # 7. 5-NN
        # 8. 10-NN
        # 9. Naive Bayes
        # 10. SGD
        # 11. AdaHERF
    
    Returns
    -------
    (str, list)
        str: The name of the type of classifier
        list: [Accuracy, Sensitivity, Specificity, ROC] classification results
    """  
    
    # Classifier type.
    if classifier_type == 1:
        tclas= 'Decision Tree'
    elif classifier_type == 2:
        tclas= 'Random Forest'
    elif classifier_type == 3:
        tclas= 'SVM (RBF)'
    elif classifier_type == 4:
        tclas= 'SVM (Linear)'
    elif classifier_type == 5:
        tclas= '1-NN'
    elif classifier_type == 6:
        tclas= '3-NN'
    elif classifier_type == 7:
        tclas= '5-NN'
    elif classifier_type == 8:
        tclas= '10-NN'
    elif classifier_type == 9:
        tclas= 'Naive Bayes'
    elif classifier_type == 10:
        tclas = 'SGD'
    elif classifier_type == 11:
        tclas = 'AdaHERF'
    else:
        tclas= 'AdaHERF'

    # Cogemos los controles.
    crl=group_df = df.ix[control_index]
    # Cogemos los pacientes.
    pat=group_df = df.ix[patient_index]
    
    X=pd.concat([crl, pat]).values
    Y=[0]*len(crl)
    Y.extend([1]*len(pat))

    res = make_classification(np.array(X),np.array(Y),classifier_type)
    
    return tclas,res


def perform_experiments(df, control_index, patient_index):
    """
    Given a DataFrame, control and patient index, we perform Leave One Out classification
    experiments comparing CART, Random Forest, SVM(RBF) and SVM(Linear). We calculate 
    the Accuracy, Sensitivity,  Specificity and AUR and we return in a DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with data (n_samples, n_features)
    control_index: pd.DataFrame
        Control index DataFrame containing True/False values. 
    patient_index: pd.DataFrame
        Patient index DataFrame containing True/False values. 
    
    Returns
    -------
    pd.DataFrame
        Pandas DataFrame including accuracy, specificity, sensitivity and AUR results for 4 classifiers.
    """  

    index = []
    index.extend(['Accuracy', 'Sensitivity', 'Specificity', 'Area Under Roc Curve'])
    columns = []

    classifier_type_range = range(1,5)
    res_mat = np.zeros((4,len(classifier_type_range)))

    for classifier_type in classifier_type_range:

        # We return row_name, col_name, Accuracy, Sensitivity, Specificity and ROC
        rexp = param_experiments(df, control_index, patient_index, classifier_type)
        columns.extend([rexp[0]])

        # Accuracy
        res_mat[0,classifier_type-1] = rexp[1][0]
        # Sensitivity
        res_mat[1,classifier_type-1] = rexp[1][1]
        # Specificity
        res_mat[2,classifier_type-1] = rexp[1][2]
        # Area Under Roc Curve
        res_mat[3,classifier_type-1] = rexp[1][3]

    results = pd.DataFrame(res_mat,index,columns)
        
    return results
