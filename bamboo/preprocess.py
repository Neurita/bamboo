"""Preprocess: Utilities to preprocess pandas DataFrame input data.
"""
# Author: Borja Ayerdi <ayerdi.borja@gmail.com>
# License: BSD 3 clause
# Copyright: UPV/EHU


def convert_to_float(df, var_list):
    """
    Converts a DataFrame selected columns from any format to float. If is not a number we put NaN.
    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame
    var_list: List
        List of variables to filter and convert.

    Returns
    -------
    df: pd.DataFrame
        Pandas DataFrame with all selected columns converted to float type.

    """
    for j in var_list:
        column = df[j]
        for i in column.index:
            val = column[i]
            try:
                val = float(val)
            except ValueError:
                df.ix[i,j] = 'NaN'
                
    # We convert all columns to float.          
    df = df.astype(float)
                
    return df
