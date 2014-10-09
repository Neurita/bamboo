"""Plot: Plotting utilities for pandas DataFrame.
"""
# Author: Borja Ayerdi <ayerdi.borja@gmail.com>
# License: BSD 3 clause
# Copyright: UPV/EHU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .features import get_feature_importance

def plot_feature_importances(df, var_names, control_index, patient_index, plot_title='Example'):
    """
    Given a pandas DataFrame, variable names, control and patient index and plot
    title we calculate feature importances for all selected variables, we plot
    them and we return the figure. We also save a *.png figure in the working dir.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame
    var_names: List
        List of variable names to get and plot feature importances.
    control_index: pd.DataFrame
        Control index DataFrame containing True/False values. 
    patient_index: pd.DataFrame
        Patient index DataFrame containing True/False values. 
   plot_title: String, optional (default='Example')
        Name of the figure.

    Returns
    -------
    plt.figure
        Plot figure with the feature importance results.

    """

    # Select controls.
    crl = df.ix[control_index]
    # Select patients.
    pat = df.ix[patient_index]

    X=pd.concat([crl, pat]).values
    Y=[0]*len(crl)
    Y.extend([1]*len(pat))
        
    row = get_feature_importance(np.array(X),np.array(Y))
        
    indices = np.argsort(row)
    indices = np.array(indices)
    indices = indices.flatten()
    indices = indices[::-1]

    if isinstance(var_names[0],str):
        var_names = [w.decode("ascii", "ignore") for w in var_names]

    var_names = np.array(var_names)
    mm = np.array(row[indices])
    mm = mm.flatten()
        
    # We only plot the first num_vars_plot values.
    num_vars_plot = 20
    ind_plot = indices[0:num_vars_plot]

    plt.rc('font',family='serif', size=20)
    plt.rc('legend', fontsize=10)

    fig = plt.figure(figsize=(20.5,10),dpi=300)
    plt.title(plot_title +" | Feature importances")
    plt.ylabel('Feature importances')
    plt.bar(range(ind_plot.shape[0]), mm[0:num_vars_plot], color="r", align="center")
    plt.xticks(range(var_names[ind_plot].shape[0]), var_names[ind_plot], rotation=90)
    plt.xlim([-1, ind_plot.shape[0]])
    plt.tight_layout()
    plt.show()

    fig_name = 'FI_'+ str(plot_title) +'.png'
    fig.savefig(fig_name)

    return fig
