from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from typing import Dict
from sklearn.metrics import auc

from src.assessment import calc_uncertainty_regection_curve, f_beta_metrics, get_performance_metric

## EVALUATION PLOTS ## 

def plot_learning_curve(history: Dict, metric: str, label: str = None, ylims: tuple = None):
    """
    Creates the learning curves of the requested metric
    :param history: A dictionary of training histories of the individual ensemble members. History
     is the attribute of the history object produced during a keras model training that it is a record of training
     loss values and metrics values at successive epochs, as well as validation loss values and validation metrics
     values
    :param metric: Metric to be plotted
    :param label: X-axis label
    :param ylims: Y-axis limits
    :return:
    """
    plt.figure(figsize=(6, 4.5))
    for num, (k, hs) in enumerate(history.items()):
        plt.plot(np.arange(1, len(hs[metric]) + 1, 1) + 1,
                 hs[metric],
                 color=f"C{num}",
                 label=f"M {k} - train")
        plt.plot(np.arange(1, len(hs[metric]) + 1, 1) + 1,
                 hs[f"val_{metric}"],
                 color=f"C{num}",
                 linestyle="--",
                 label=f"M {k} - dev in")
    if len(history.keys()) > 1:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc="upper right")
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.grid()
    plt.ylim(ylims)
    plt.show()
    plt.close()


def get_comparison_error_retention_plot(error, uncertainty):
    """
    Creates an error retention plot for the three error ranking types: "random", "ordered" and "optimal"
    :param error: Per sample errors - array [n_samples]
    :param uncertainty: Uncertainties associated with each prediction. array [n_samples]
    :return:
    """
    for rank_type in ["random", "ordered", "optimal"]:
        rejection_mse = calc_uncertainty_regection_curve(errors=error,
                                                         uncertainty=uncertainty,
                                                         group_by_uncertainty=False,
                                                         rank_type=rank_type)
        retention_mse = rejection_mse[::-1]
        retention_fractions = np.linspace(0, 1, len(retention_mse))
        roc_auc = auc(x=retention_fractions[::-1], y=retention_mse)

        label = "Model" if rank_type == "ordered" else rank_type.capitalize()
        plt.plot(retention_fractions, retention_mse, label=f"{label} R-AUC:" + "{:.1e}".format(roc_auc))


def error_retention_plot_allsets(uncertainties, sqr_errors, sets_names):
    """ Plot the error retention plot for each subset of data. 
        sets_name: names of subsets of data = keys of dicts uncertainties and sqr_errors """
    plt.figure(figsize=(16, 5))
    plt.subplots_adjust(wspace=.3, hspace=.4)
    for i, partition in enumerate(sets_names):
        plt.subplot(1, 3, i + 1)
        get_comparison_error_retention_plot(error=sqr_errors[partition],
                                            uncertainty=uncertainties[partition]["tvar"])
        plt.ylabel('MSE(W)')
        plt.xlabel("Retention Fraction")
        plt.legend(loc="upper left", prop={"size": 12})
        plt.title(partition, pad=25)
        # plt.ylim(-100, 2300)
        plt.grid()
    plt.show()
    plt.close()


def get_comparison_f1_retention_plot(error, uncertainty, threshold):
    """
    Creates a F1-score retention plot for the three error ranking types: "random", "ordered" and "optimal"
    :param error: Per sample errors - array [n_samples]
    :param uncertainty: Uncertainties associated with each prediction. array [n_samples]
    :param threshold: The error threshold below which we consider the prediction acceptable
    :return:
    """
    for rank_type in ["random", "ordered", "optimal"]:
        f_auc, f95, retention_f1 = f_beta_metrics(errors=error,
                                                  uncertainty=uncertainty,
                                                  threshold=threshold,
                                                  beta=1.0,
                                                  rank_type=rank_type)
        #         print(f"{rank_type}: F1 score at 95% retention: ", f95)
        retention_fractions = np.linspace(0, 1, len(retention_f1))

        label = "Model" if rank_type == "ordered" else rank_type.capitalize()
        plt.plot(retention_fractions, retention_f1, label=f"{label} F1-AUC:{np.round(f_auc, 3)}")

def f1_retention_plot_allsets (uncertainties, sqr_errors, sets_names, thresh):
    """ F1-score retention plot for each subset of data named in sets_names. """
    plt.figure(figsize=(16, 5))
    plt.subplots_adjust(wspace=.3, hspace=.4)
    for i, partition in enumerate(sets_names):
        plt.subplot(1, 3, i + 1)
        get_comparison_f1_retention_plot(error=sqr_errors[partition], uncertainty=uncertainties[partition]["tvar"], threshold=thresh)
        plt.legend(loc="lower right", prop={"size": 12})
        plt.ylabel('F1')
        plt.xlabel("Retention Fraction")
        plt.title(partition, pad=25)
        # plt.ylim(-0.1, 1.2)
        plt.grid()
    plt.show()
    plt.close()


def table_errors(predictions, datasets, target):
    """ Plots a table for the rmse, mae and mape of the predictions according to target. """
    table = [["Data", "RMSE Ens. (KW)", "MAE Ens. (KW)", "MAPE (%)"]]
    for k in datasets.keys():
        ens_predictions = np.squeeze(np.mean(predictions[k]["denorm"][:, :, 0], axis=0))
        rmse_metric = get_performance_metric(y_pred=ens_predictions, y_true=datasets[k][target], metric="rmse")
        mae_metric = get_performance_metric(y_pred=ens_predictions, y_true=datasets[k][target], metric="mae")
        mape_metric = get_performance_metric(y_pred=ens_predictions, y_true=datasets[k][target], metric="mape")
        results = [k,
                np.round(rmse_metric, 0),
                np.round(mae_metric, 0),
                np.round(100 * mape_metric, 2)]
        table.append(results)
    print("Classic metrics")
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


def table_auc(sets_names, sqr_errors, uncertainties, thresh):
    """ Plots a table for R-AUC, F1-AUC and F1 at 95% with the errors and uncertainty of predictions. """
    table = [["Data", "R-AUC", "F1-AUC", "F1 @ 95 %"]]
    for k in sets_names:
        r = sqr_errors[k]
        var = uncertainties[k]["tvar"]

        rejection_mse = calc_uncertainty_regection_curve(errors=r, uncertainty=var, group_by_uncertainty=False)
        retention_mse = rejection_mse[::-1]
        retention_fractions = np.linspace(0, 1, len(retention_mse))
        roc_auc = auc(x=retention_fractions[::-1], y=retention_mse)

        f_auc, f95, retention_f1 = f_beta_metrics(errors=r, uncertainty=var, threshold=thresh, beta=1.0)
        results = [k,
                roc_auc,
                np.round(f_auc, 3),
                np.round(f95, 3)]
        table.append(results)
    print("AUCs")
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


def plot_flight_pred(datasets, flight, pred_test_moy, test_name):
    """ Plot one flight and its predictions given by the model, as well as real data for those points """
    figsize=(25,15)
    # plot line for data of this flight
    dataset_complete = pd.concat([datasets[k] for k in datasets.keys()])
    dataset_complete.loc[dataset_complete.flight == flight, "power_smoothed"].sort_index().plot(marker="", figsize=figsize, label="real data")
    # plot markers of test labels
    data_test = datasets[test_name]
    ex_test = data_test.loc[data_test.flight == flight]
    ex_test["power"].sort_index().plot(marker=".", figsize=figsize, ls='', label="labels", color="r")
    data_test.loc[data_test.flight==flight, "power"].sort_index().plot(marker=".", figsize=figsize, ls='', label="labels", color="r")
    # plot predictions with uncertainty
    yerr = pred_test_moy.loc[ex_test.index, "std"].sort_index()
    moy = pred_test_moy.loc[ex_test.index,"mean"].sort_index()
    moy.plot(marker=".", figsize=figsize, ls='-', label="mean prediction", color="black")
    plt.fill_between(ex_test.sort_index().index, moy - yerr, moy + yerr, color="black", alpha=0.2)

    # plt.xlim(250250, 250600)
    plt.legend()
    plt.grid()
    plt.show()