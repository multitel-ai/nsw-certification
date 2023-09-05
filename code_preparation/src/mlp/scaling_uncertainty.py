import json
import os
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
from typing import Dict

from src.assessment import get_model_errors


## SCALING DATA ##

def store_stats(train_data, model_dir):
    """ Computes means and sts of each feature and stores them in given directory. """
    means = train_data.mean()
    stds = train_data.std()
    # wind angle avec stats circulaires 
    means["wind_angle"] = circmean(train_data, high=360)
    means["wind_angle"] = circstd(train_data, high=360)

    store_path = os.path.join(model_dir, 'stats')
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    with open(os.path.join(store_path, 'means.json'), 'w') as f:
        json.dump(means.to_dict(), f, indent=4)
    with open(os.path.join(store_path, 'stds.json'), 'w') as f:
        json.dump(stds.to_dict(), f, indent=4)

    return means, stds


def data_normalization(data: pd.DataFrame, means: pd.Series, stds: pd.Series, time_max) -> pd.DataFrame:
    """
    Data scaling process:
        1. Standardization scaling for all features except timeSinceDryDock
        2. Zero max scaling of the time feature considering max value=8 years
    :param data: The dataframe to scale
    :param means: Mean values of the features to be scaled
    :param stds: Standard deviation values of the features to be scaled
    :return: Scaled dataset
    """
    features_norm = data.columns.difference(['time'])

    data_norm = (data[features_norm] - means[features_norm]) / stds[features_norm]
    data_norm["time"] = data["time"] / time_max
    # data_norm["flight"] = data["flight"]

    return data_norm


def denorm_prediction(preds_norm, target_mean, target_std):
    """
    Denormalize model predictions. This is the inverse transformation of the data_normalization function
    :param preds_norm: Normalized predictions. An array with size [number of models x number of samples x 2], where 2
    accounts for [mean, variance]
    :param target_mean: Target mean value to be used
    :param target_std: Target standard deviation to be used
    :return: Denormalized predictions
    """
    preds_denorm = preds_norm.copy()
    preds_denorm[:, :, 0] = (preds_denorm[:, :, 0] * target_std) + target_mean
    preds_denorm[:, :, 1] = preds_denorm[:, :, 1] * np.square(target_std)
    return preds_denorm


## UNCERTAINTY ##

def epkl_reg(preds: np.array, epsilon=1e-10) -> np.array:
    """
    Taken from here: https://github.com/yandex-research/shifts.
    Modification/additions:
    Rename vars variable to vars_.
    :param: preds: array [n_models, n_samples, 2] - mean and var along last axis.
    """
    means = preds[:, :, 0]
    vars_ = preds[:, :, 1] + epsilon
    logvars = np.log(vars_)

    avg_means = np.mean(means, axis=0)
    avg_second_moments = np.mean(means * means + vars_, axis=0)

    inv_vars = 1.0 / vars_
    avg_inv_vars = np.mean(inv_vars, axis=0)
    mean_inv_var = inv_vars * means
    avg_mean_inv_var = np.mean(mean_inv_var, axis=0)
    avg_mean2_inv_var = np.mean(means * mean_inv_var + logvars, axis=0) + np.log(2 * np.pi)

    epkl = 0.5 * (avg_second_moments * avg_inv_vars - 2 * avg_means * avg_mean_inv_var + avg_mean2_inv_var)

    return epkl


def ensemble_uncertainties_regression(preds: np.array) -> Dict:
    """
    Taken from here: https://github.com/yandex-research/shifts
    :param: preds: array [n_models, n_samples, 2] - last dim is mean, var
    """
    epkl = epkl_reg(preds=preds)
    var_mean = np.var(preds[:, :, 0], axis=0)
    mean_var = np.mean(preds[:, :, 1], axis=0)

    uncertainty = {'tvar': var_mean + mean_var,
                   'mvar': mean_var,
                   'varm': var_mean,
                   'epkl': epkl}

    return uncertainty


def compute_error(predictions, datasets, target):
    """ Computes uncertainties and rmse of the predictions """
    uncertainties = {}
    for k in datasets.keys():
        uncertainties[k] = ensemble_uncertainties_regression(preds=predictions[k]["denorm"])

    sqr_errors = {}
    for k in datasets.keys():
        ens_predictions = np.squeeze(np.mean(predictions[k]["denorm"][:, :, 0], axis=0))
        power_labels = np.asarray(datasets[k][target])
        sqr_errors[k] = get_model_errors(y_pred=ens_predictions, y_true=power_labels)
    
    return uncertainties, sqr_errors