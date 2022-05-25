""" Tools that should make it easier to read and write data """

import os
import json
import sys


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


def savePlot(fig, outpath, **kwargs):
    """ Convenience wrapper for saving plots. """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, **kwargs)
