import pandas as pd

def add_deblinding(df, src_file, keys, target):
    """Combines a dataframe with trial info with a dataframe with session info

    Specifically, the indices and columns in src_file are used as values to add
    to the df dataframe. Therefore, users have to make sure that they match. 

    Purpose here is to add subject/session drug information to single trial data

    Parameters
    ----------
    df : Pandas Dataframe
        Data where the target column should be added to
    src_file : String | Pathlike
        file to read target information from. Indices and column names are the
        ones used to index the correct cells in the df
    keys : list 
        names of columns in df that should match indices and columns in src_file
    target : String
        Name of column that receives the additional information
    """
    # read src_file
    deblind = pd.read_csv(src_file, index_col=0, header=None, sep='\t')

    # loop over unique values and add the info in src_file to the df
    for sub in deblind.index:
        for ses in deblind.columns:
            df.loc[(df[keys[0]] == sub) & (df[keys[1]] == ses), target] = deblind.loc[sub, ses]
    return df