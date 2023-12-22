import numpy as np
from pandas import (DataFrame, MultiIndex)

def setup_sub_ses_df(n_sub, n_ses, columns):
    """Creates empty DataFrame with (n_sub, n_ses) index and columns."""
    arrays = [np.repeat(np.arange(1, n_sub + 1), n_ses),
              np.tile(list(range(1, n_ses + 1)), n_sub)]
    idx = MultiIndex.from_arrays(arrays, names=('subject', 'session'))
    return DataFrame(columns=columns, index=idx)