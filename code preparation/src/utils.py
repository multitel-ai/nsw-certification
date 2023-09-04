from ast import Dict
import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns



## DATA LOADERS ##

def read_split_csv(folder_path):
    """Load csv files in folder path in a dictionnary of pandas dataframes"""
    datasets = dict()
    for f in os.listdir(folder_path):
        print(f)
        datasets[f[:-4]] = pd.read_csv(os.path.join(folder_path, f), index_col=0)
    return datasets

def read_complete(folder_path):
    """Lit les datasets du folder path et renvoie la concatenation"""
    datasets = read_split_csv(folder_path)
    # create one df with all 
    dataset_complete = pd.concat([datasets[key] for key in datasets.keys()]).sort_index()
    return dataset_complete


## VISUALISATION DES DONNÉES ##

def violin_plot(x: pd.Series, y: pd.Series, xlabel: str):
    """
    Creates a violin plot
    :param x: Input for x-axis
    :param y: Input for y-axis
    :param xlabel: Label to be used for the x-axis
    :return:
    """
    plt.figure(figsize=(7, 6))
    sns.violinplot(x=x, y=y, linewidth=1, scale="width", palette="pastel")
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.grid(axis="x")
    plt.show()
    plt.close()


def get_density_plots_matrix(data: pd.DataFrame, types_under_study, features_under_study,
                             num_rows=4, num_cols=3):
    """
    Creates a matrix (num_rows, num_cols) of density plots of the features_under_study for the datasets defined by the
    type labels in types_under_study
    :param data: A pandas dataframe. It is required to have a column named 'type' that is a categorical variable
    indicating the type of individual datasets
    :param types_under_study: A list of the type labels to be plotted
    :param features_under_study: A list of features for which density plots are to be produced
    :param colors_mapping: A dictionary the maps each dataset type label at a given color
    :param num_rows: Number of rows of the matrix
    :param num_cols: Number of columns of the matrix
    :return:
    """
    plt.figure(figsize=(12, 14))
    plt.subplots_adjust(wspace=.4, hspace=.4)

    for i, f in enumerate(features_under_study):
        plt.subplot(num_rows, num_cols, i + 1)
        for t in types_under_study:
            mask = data["type"] == t
            data[mask][f].plot.kde(label=t)
            plt.xlim((-1, 1) if f == "diff_speed_overground" else None)
            plt.title(f)
    plt.show()
    plt.close()