"""
Some nifty utils

author: William Tong (wtong@g.harvard.edu)
"""

import numpy as np

def new_seed(): return np.random.randint(1, np.iinfo(np.int32).max)