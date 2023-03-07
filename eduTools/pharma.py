import pandas as pd
import glob
import os
from eduTools.io import read_json

def add_deblinding(df, drug_df, keys, target):
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
    # loop over unique values and add the info in src_file to the df
    for (sub, ses) in drug_df.index:
        df.loc[(sub, ses), target] = drug_df.loc[(sub, ses), 'drug_label']
    return df


def extract_drug_info(search_dir, search_pattern):
    """looks for a drug field in collection"""

    df = pd.DataFrame(columns=['drug'], index=pd.MultiIndex(levels=[[], []], 
                                                            codes=[[], []]))
    glob_files = sorted(glob.glob(search_dir + os.sep + search_pattern))
    for file in glob_files:
        json = read_json(file)
        sub = json['subject']
        ses = json['session']
        df.loc[(sub, ses), 'drug'] = json['drug']
    return df