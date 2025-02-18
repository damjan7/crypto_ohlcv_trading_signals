"""
Loads single features  and combines them into a single signals.
Since we are creating signals in this file, we need to create the feature dataset for all pairs of interest.

TODO:
- normalize features

first easy method:
- create z-scores and simply combine via averaging. Then use simple sorting portfolios
- implement something weighting rules according to place in the z-score distribution
"""

import data_processor as dp
import pandas as pd
from typing import List


# idea: write base data class that holds all the data
# upon init; automatically creates dictionary, cross sectional ds, and everything else

# then can easily re-use for different signals and don't need to rewrite all the function calls
# can also implement an @abstractmethod 'create_signal()' and some more stuff



def test_signal_1(pairs: List[str], df: pd.DataFrame): 
    """
    Example
    24h Vol. has to be higher than 7 day avg
    """

    feature_dict, feature_names = create_feature_ds_dict_for_pairs(pairs)
