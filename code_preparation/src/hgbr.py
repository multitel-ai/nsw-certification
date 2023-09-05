import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def plot_loss(hgbr, log_scale=False):
    """ Plot the training loss of model hgbr with log scale in y axis or not. Returns the axes."""
    _, ax = plt.subplots()
    ax.plot(-hgbr.train_score_, label='Train')
    ax.plot(-hgbr.validation_score_, label="Validation")
    if log_scale:
        ax.set_yscale("log") 
    ax.set(xlabel="Boosting iterations (trees)", ylabel="Negative loss (squared error)", title="Training monitoring")
    ax.set(ylabel="mse")
    plt.legend()
    return ax


def histgradregressor(X_train, y_train, X_test, y_test, seed, max_iter=200, verbose=False, regularization=0, learning_rate=0.1, log_scale=False):
    """ Train HGBR model with given train data. Plot the training loss. 
    Evaluate the model with the given test data : MSE computation printed for the final model and plotted for each epoch (group of trees) with the loss curve. 
    Returns the trained model.
    """
    # entrainement 
    tic = time.time()
    hgbr = HistGradientBoostingRegressor(loss="squared_error", verbose=verbose, random_state=seed, 
                                         max_iter=max_iter, l2_regularization=regularization, learning_rate=learning_rate)
    hgbr = hgbr.fit(X_train, y_train)
    duree_sec = time.time() - tic
    duree_min = int(duree_sec // 60)
    duree_sec_restantes = int(duree_sec % 60)
    print("La durée d'exécution est de", duree_min, "minutes et", duree_sec_restantes, "secondes.")
    # plot la loss d'entrainement 
    ax = plot_loss(hgbr, log_scale)
    # Faire des prédictions sur X_test
    y_pred = hgbr.predict(X_test)
    # Évaluer les performances du modèle
    mse = mean_squared_error(y_test, y_pred)
    print("Root of Mean Squared Error:", np.sqrt(mse))
    print("R² score :", 100*hgbr.score(X_test, y_test))
    # Faire des prédictions graduées à chaque étape de l'entraînement
    predictions = [pred for pred in hgbr.staged_predict(X_test)]
    # Calculer l'erreur quadratique moyenne (MSE) à chaque étape
    mseL = [mean_squared_error(y_test, pred) for pred in predictions]
    ax.plot(mseL, label="Test", color="green")
    ax.set(ylabel="mse")
    plt.legend()
    plt.show()

    return hgbr


def cross_validate_group(X, y, groups, seed, num_folds=5, scoring='r2', verbose=False, max_iter=300, regularization=0.2, learning_rate=0.1):
    """ Cross validation grouped by flight number. Returns the scores according to chosen scoring."""
    group_kfold = GroupKFold(n_splits=num_folds)
    model = HistGradientBoostingRegressor(loss="squared_error", verbose=verbose, random_state=seed, max_iter=max_iter, 
                                          l2_regularization=regularization, learning_rate=learning_rate)  
    scores = cross_val_score(model, X, y, cv=group_kfold, groups=groups, scoring=scoring, n_jobs=-1, verbose=verbose)
    if scoring == "r2":
        scores *= 100
        print("Score in percentage")
    print("Median of", scoring)
    print(np.round(np.median(scores), 2), "+-", np.round(np.std(scores),2))
    plt.figure(figsize=(3,3))
    plt.boxplot(scores)
    plt.show()
    return scores


def plot_loss_cv(X_used, y, groups, seed, n_splits=10,  max_iter=300, regularization=1000, learning_rate=0.1):
    """ Plot the loss curve for the first grouped fold with the chosen model parameters. Returns the model trained"""
    gkf = GroupKFold(n_splits=n_splits)
    y_used = y.loc[X_used.index]
    for train_index, test_index in gkf.split(X_used, y_used, groups=groups):
        X_train = X_used.iloc[train_index, :]
        y_train = y_used.iloc[train_index]
        X_test = X_used.iloc[test_index, :]
        y_test = y_used.iloc[test_index]
        hgbr = histgradregressor(X_train, y_train, X_test, y_test, seed, max_iter=max_iter, regularization=regularization, learning_rate=learning_rate)
        break 
    return hgbr


def train_evaluate(X_used, y, groups, seed, max_iter=150, regularization=1000, learning_rate=0.1, n_splits=5):
    """ Function to train and evaluate with different methods given parameters with given data. 
    Split shuffled data and plot loss ; then cross validation with grouped data and 2 metrics : rmse and r².
    """
    # split random pour tester les paramètres et regarder la loss 
    X_train, X_test, y_train, y_test = train_test_split(X_used, y[X_used.index], random_state=seed)
    # train shuffled data and plot loss 
    print("==Shuffled data==")
    histgradregressor(X_train, y_train, X_test, y_test, seed, max_iter=max_iter, regularization=regularization, learning_rate=learning_rate)
    # plot loss for first fold of CV 
    print("==Grouped by flight==")
    plot_loss_cv(X_used, y, groups, seed, n_splits=n_splits, max_iter=max_iter, regularization=regularization, learning_rate=learning_rate)
    # CV complete
    cross_validate_group(X_used, y[X_used.index], groups=groups, seed=seed, scoring="neg_root_mean_squared_error", num_folds=n_splits, max_iter=max_iter, 
                         regularization=regularization, learning_rate=learning_rate)
    cross_validate_group(X_used, y[X_used.index], groups=groups, seed=seed, scoring="r2", num_folds=n_splits, max_iter=max_iter, 
                         regularization=regularization, learning_rate=learning_rate)


def scatter_predictions(X_train, y_train, seed, max_iter=200, regularization=1000, learning_rate=0.1):
    """ Scatter plot of predictions of model according to real data, on training data. """
    hgbr = HistGradientBoostingRegressor(loss="squared_error", random_state=seed, 
                                            max_iter=max_iter, l2_regularization=regularization, learning_rate=learning_rate)
    hgbr = hgbr.fit(X_train, y_train)
    y_pred = hgbr.predict(X_train)
    _, ax = plt.subplots()
    ax.scatter(y_train, y_pred, color='blue', label='Prédictions', alpha=0.25, marker=".")
    ax.plot(y_train, y_train, color='red', label='x=y')
    ax.set_xlabel('Valeurs Réelles')
    ax.set_ylabel('Prédictions')
    ax.set_title('Scatter Plot des Prédictions d\'un Modèle de Régression')
    ax.legend()
    plt.show()