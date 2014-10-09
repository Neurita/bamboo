"""Stats: Utilities to calculate stats over pandas DataFrame input data.
"""
# Author: Borja Ayerdi <ayerdi.borja@gmail.com>
# License: BSD 3 clause
# Copyright: UPV/EHU

import numpy as np
import pandas as pd
from scipy import stats


def do_ttest(df, control_index, patient_index, ttest_var_names, only_significative=False):
    """
    Given a DataFrame, Control and Patiend indexes and some variables performs a Welch's t-test assuming unequal variances.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame
    control_index: pd.DataFrame
        Control index DataFrame containing True/False values. 
    patient_index: pd.DataFrame
        Patient index DataFrame containing True/False values. 
    ttest_var_names: List
        List of variable names to perform ttest.
    only_significative: boolean, optional
        If we want to return only p-values results of variables with significative diference or all p-values

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with p-values of performed statistical tests.

    """
    # http://iaingallagher.tumblr.com/post/50980987285/t-tests-in-python
    d = np.zeros((len(ttest_var_names), 1))
    idx_row = 0

    group1_df = df.ix[control_index]
    group2_df = df.ix[patient_index]
        
    for v in ttest_var_names:
        group1_var = group1_df[v].dropna()
        group2_var = group2_df[v].dropna()
        group1_values = group1_var.values
        group2_values = group2_var.values

        # assuming unequal population variances
        two_sample_diff_var = stats.ttest_ind(group1_values, group2_values, equal_var=False)

        d[idx_row, 0] = two_sample_diff_var[1]
        idx_row = idx_row + 1

    df = pd.DataFrame(d, index=ttest_var_names, columns={'P-Value'})

    # Return only variables with significative difference. 
    if only_significative:
        df = df[df<0.01].dropna()
       
    return df


def var_group_means(df, var_names, control_index, patient_index, group1_name='Controls', group2_name='Patients'):
    """
    Given a DataFrame, variable names list, control and patiend index and groups names performs mean and var values for each variable and group.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame
    var_names: List
        List of variable names to perform means.
    control_index: pd.DataFrame
        Control index DataFrame containing True/False values. 
    patient_index: pd.DataFrame
        Patient index DataFrame containing True/False values. 
    group1_name: String, optional (default='Controls')
        First group name.
    group2_name: String, optional (default='Patients')
        Second group name.

    Returns
    -------
    pd.DataFrame
        DataFrame with means and variance for each variable grouped by group type.

    """
    groups = (group1_name+' (mean)', group1_name+' (var)', group2_name+' (mean)', group2_name+' (var)')
    num_Rows = len(var_names)
    num_Cols = len(groups)
    d = np.zeros((num_Rows,num_Cols))
    idx_col = 0

    group_ctr = df.ix[control_index]
    group_pat = df.ix[patient_index]

    idx_row = 0
    for v in var_names:
        group_var = group_ctr[v].dropna()
        d[idx_row, 0] = group_var.mean()
        d[idx_row, 1] = group_var.var()
        group_var = group_pat[v].dropna()
        d[idx_row, 2] = group_var.mean()
        d[idx_row, 3] = group_var.mean()
        idx_row = idx_row + 1

    df = pd.DataFrame(d, columns=groups, index=var_names)
    return df

