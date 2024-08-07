import os

# This file allows to configure where to save data, results, plots etc.
class CustomPaths:
    # path where downloaded data sets will be saved
    data_path = 'data'
    # path where benchmark results will be saved
    results_path = 'results'
    # path where plots and tables will be saved
    plots_path = 'plots'
    # path where benchmark results can be cached in a more efficient format such that they load faster
    cache_path = 'cache'


def get_data_path():
    if not os.path.exists(CustomPaths.data_path):
        os.makedirs(CustomPaths.data_path)
    return CustomPaths.data_path


def get_results_path():
    if not os.path.exists(CustomPaths.results_path):
        os.makedirs(CustomPaths.results_path)
    return CustomPaths.results_path


def get_plots_path():
    if not os.path.exists(CustomPaths.plots_path):
        os.makedirs(CustomPaths.plots_path)
    return CustomPaths.plots_path


def get_cache_path():
    if not os.path.exists(CustomPaths.cache_path):
        os.makedirs(CustomPaths.cache_path)
    return CustomPaths.cache_path

