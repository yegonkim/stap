from .feature_functions import *
FEATURE_FUNC = {
    # "zcov": get_features_zcov, 
    "ycov": get_features_ycov, 
    # "z": get_features_z, 
    # "y": get_features_y,
    # "p": get_features_p,
    # "x": get_features_x, 
    # "zmean": get_features_zmean, 
    # "ymean": get_features_ymean
    }

from .acquisition_functions import *
ACQUISITION_FUNCTIONS = {
    "constant": constant,
    "entropy": entropy,
    "variance": variance,
    # "kmeans": kmeans,
    "bait": bait,
    # "coreset": coreset,
    }
