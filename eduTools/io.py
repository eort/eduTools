""" Tools that should make it easier to read and write data """

import os
import json
import sys
from IPython import embed as shell
import glob
import pandas as pd
import logging

def read_json(fname):
    """Save reading of json files

    Parameters
    ----------
    fname : String | PosixPath
        file name that should be used
    """
    try:
        with open(fname, 'r') as f:    
            info = json.load(f)
    except FileNotFoundError:
        error(f"File {fname} was not found! Stop procedure")
        sys.exit(1)
    return info

def write_json(fname, data=None):
    """ Save reading and writing of json files

    Parameters
    ----------
    fname : String | PosixPath
        file name that should be used
    data : Dict
        data to be written
    """

    # first check whether data exists
    assert data is not None
    # then make sure that an existing file wont be overwritten
    try:
        with open(fname, 'r') as f:    
            temp_data = json.load(f)
    except FileNotFoundError:
        temp_data = {}

    # do some patching in case we have nested dictionaries
    for k, v in data.items():
        if type(v) == dict:
            try:
                temp_data[k].update(data[k])
            except KeyError:
                temp_data[k] = v
        else:
            temp_data.update({k:v})
    # and write to file
    with open(fname, 'w') as f:
        json.dump(temp_data, f, sort_keys=True, indent=4)


def save_plot(fig, outpath, **kwargs):
    """ Convenience wrapper for saving plots. """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, **kwargs)


def concatenate_text_files(search_dir, search_pattern, sep='\t'):
    """Combines all files that match the pattern in a single dataframe

    Parameters
    ----------
    search_dir : Path-like | String
        directory where the to-be-concatenated files will be search for
    search_pattern : String
        Glob pattern that will be used to detect files
    sep : String
        what delimiting values are used in the files to be concatenated
    """
    # find files
    glob_files = sorted(glob.glob(search_dir + os.sep + search_pattern))
    logging.info(f"Reading data from {search_dir}/{search_pattern}.")
    # read files
    infiles = [pd.read_csv(f, sep=sep) for f in glob_files]
    
    return pd.concat(infiles, axis=0, ignore_index=True)



