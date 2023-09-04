import json
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import tqdm
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from src.mlp.scaling_uncertainty import denorm_prediction



## DATA CLASS ##

class VPowerDataset(Dataset):
    """ Classe pour les données pour l'entrainement """
    def __init__(self, data, input_features, target):
        self.norm_df = data
        self.items = self.norm_df[input_features].values
        self.labels = self.norm_df[target].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.items[idx]
        label = self.labels[idx]
        return {"Item": item, "Power": label}


## MODEL ##

class ProbMCdropoutDNN(nn.Module):
    """
    Monte Carlo (MC) dropout neural network with 2 hidden layers.
    """
    def __init__(self, input_size, hidden_size_1=50, hidden_size_2=20, dropout=0.005):
        super(ProbMCdropoutDNN, self).__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size_1)
        self.linear2 = nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2)
        self.linear3 = nn.Linear(in_features=hidden_size_2, out_features=2)
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # first linear layer
        x = self.linear1(x)
        x = self.softplus(x)
        x = self.dropout(x)
        # second linear layer
        x = self.linear2(x)
        x = self.softplus(x)
        x = self.dropout(x)
        # output
        x = self.linear3(x)
        return torch.distributions.Normal(
            loc=x[:, 0:1].squeeze(),
            scale=self.softplus(x[:, 1:2].squeeze()).add(other=1e-6)
        )

    def predict(self, x):
        distrs = self.forward(x)
        y_pred = distrs.sample()
        return y_pred


def check_metric(curr_value, best_value, not_improved, model, store_path):
    """ Checks if metric used for early stopping has improved or not : if yes, saves the new model, otherwise increments counter """
    if curr_value < best_value:
        best_value = curr_value
        torch.save(model.state_dict(), os.path.join(store_path, "best_model.pth"))
        print(f"Saved to {store_path}/best_model.pth")
        not_improved = 0
    else:
        not_improved += 1
    return best_value, not_improved


def train(model, train_dataset, val_dataset, optimizer, store_path, es_monitor, device, batch_size=32, n_epochs=80, patience=4):
    """
        Model training with early stopping.
        :param model: model
        :param train_dataset: A VPowerDataset object of the training data
        :param val_dataset: A VPowerDataset object of the validation data
        :param optimizer: Model optimizer
        :param batch_size: Batch size
        :param n_epochs: Maximum epochs to be performed
        :param es_monitor: Early stopping monitoring metric
        :param patience: Early stopping patience
        :param store_path: Path to store the model
        :return: training history dictionary
    """
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    best_mae = np.inf
    best_mse = np.inf
    not_improved = 0
    history = {
        metric: [] for metric in [
            "loss",
            "mse",
            "mae",
            "val_loss",
            "val_mse",
            "val_mae"
        ]
    }

    for epoch in range(n_epochs):
        model.train()

        epoch_losses = []
        epoch_mses = []
        epoch_maes = []
        print(f"Epoch {epoch}/{n_epochs}")

        for batch_data in train_loader:
            inputs, labels = ( 
                batch_data["Item"].float().to(device), # load them to device 
                batch_data["Power"].float().to(device)
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = -outputs.log_prob(labels).mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            y_pred = model.predict(inputs)
            epoch_mses.append(mse(y_true=labels.cpu(), y_pred=y_pred.cpu()))
            epoch_maes.append(mae(y_true=labels.cpu(), y_pred=y_pred.cpu()))
        curr_loss = sum(epoch_losses) / len(epoch_losses)
        print("Average train loss:", curr_loss)
        curr_mse = sum(epoch_mses) / len(epoch_mses)
        curr_mae = sum(epoch_maes) / len(epoch_maes)
        history["loss"].append(curr_loss)
        history["mse"].append(curr_mse)
        history["mae"].append(curr_mae)

        # Validation step -----------------------------------
        model.eval()
        val_losses = []
        val_mses = []
        val_maes = []
        with torch.no_grad():
            for val_batch_data in val_loader:
                val_inputs, val_labels = (
                    val_batch_data["Item"].float().to(device),
                    val_batch_data["Power"].float().to(device)
                )
                val_outputs = model(val_inputs)

                val_loss = -val_outputs.log_prob(val_labels).mean()
                val_losses.append(val_loss.item())

                y_pred = model.predict(val_inputs)
                val_mses.append(mse(y_true=val_labels.cpu(), y_pred=y_pred.cpu()))
                val_maes.append(mae(y_true=val_labels.cpu(), y_pred=y_pred.cpu()))
        curr_val_loss = sum(val_losses) / len(val_losses)
        curr_val_mse = sum(val_mses) / len(val_mses)
        curr_val_mae = sum(val_maes) / len(val_maes)
        print("Average val loss:", curr_val_loss)
        print(f"Val MSE:", curr_val_mse)
        print(f"Val MAE:", curr_val_mae)

        history["val_loss"].append(curr_val_loss)
        history["val_mse"].append(curr_val_mse)
        history["val_mae"].append(curr_val_mae)

        # Early stopping monitoring val metrics  ---------------------------------
        if es_monitor == "val_mae":
            best_mae, not_improved = check_metric(curr_value=curr_val_mae, best_value=best_mae,
                                                  not_improved=not_improved, model=model, store_path=store_path)
        elif es_monitor == "val_mse":
            best_mse, not_improved = check_metric(curr_value=curr_val_mse, best_value=best_mse,
                                                  not_improved=not_improved, model=model, store_path=store_path)
        if not_improved == patience:
            print("Early stopping")
            break
    return history


def train_save(imin, imax, model_dir, device, train_ds, val_ds, 
               input_f, hiddens=(50,20), dropout=5e-3, # paramètres du modèle
               lr=1e-2, batch_size=1024, n_epochs=150, es_monitor="val_mae", patience=15): # paramètres de l'entrainement
    """ Trains a certain number (imin -> imax) of models on device and saves them in model_dir. Prints the best epoch of each model and its metrics.
    Parameters needed are the training and validation set, model parameters  and training parameters. 
    Returns a dictionnary with the trained models. """
    models_hs = {}
    for i in range(imin, imax+1):
        store_path = os.path.join(model_dir, f"member_{i}/")
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        model = ProbMCdropoutDNN(input_size=len(input_f),
                                hidden_size_1=hiddens[0],
                                hidden_size_2=hiddens[1],
                                dropout=dropout).to(device) # load on gpu or cpu 
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        history = train(model=model,
                        train_dataset=train_ds,
                        val_dataset=val_ds,
                        optimizer=optimizer,
                        store_path=store_path,
                        device=device,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        es_monitor=es_monitor,
                        patience=patience)
        models_hs[i] = history

        with open(os.path.join(store_path, f"history.json"), 'w') as jsonfile:
            json.dump(history, jsonfile)

    for i in range(imin, imax+1):
        print(f"Modèle {i} :\n Validation loss :{np.min(models_hs[i]['val_loss'])} ; Validation MSE: {np.min(models_hs[i]['val_mse'])} ; \
                Validation MAE: {np.min(models_hs[i]['val_mae'])} ; Nombre d'époques: {np.argmin(models_hs[i][es_monitor])}")
        
    return models_hs


def load_models(model_dir, imin, imax, input_f, hiddens=(50,20), dropout=5e-3): 
    """ Load given models (imin -> imax) from model_dir. Needs model parameters. Returns a list of loaded models. """
    loaded_models = []
    for i in range(imin, imax+1):
        model = ProbMCdropoutDNN(input_size=len(input_f), hidden_size_1=hiddens[0], hidden_size_2=hiddens[1], dropout=dropout)
        load_path = os.path.join(model_dir, f"member_{i}", "best_model.pth")
        model.load_state_dict(torch.load(load_path))
        loaded_models.append(model)
    return loaded_models


## GET PREDICTIONS ## 

def get_distributions_params(model, x):
    """
    Return the parameters of the normal distributions
    :param model: The model
    :param x: The input tensor
    :return: array [num_samples x 2],
             where
             num_samples = number of rows in data_norm
             2 = [mean, std]
    """
    distrs = model.forward(x)
    return np.concatenate([distrs.loc.detach().numpy().reshape(-1, 1),
                           distrs.scale.detach().numpy().reshape(-1, 1)],
                          axis=1)

def get_ensemble_predictions(model_list, data_norm, multi_runs):
    """
    Gets the normalized predictions of an ensemble of models
    :param model_list: List of models to form an ensemble
    :param data_norm: Torch tensor of the input data
    :param multi_runs: Number of sequential runs per member of the ensemble
    :return: An array with the ensemble predictions with size [(ensemble_size * multi_runs) x num_samples x 2],
            where
            ensemble_size = number of models in models_list
            multi_runs = number of sequential runs per member of the ensemble
            num_samples = number of rows in data_norm
            2 = [mean, variance]
    """
    all_preds = []
    for member in tqdm.tqdm(range(len(model_list))):  # Iterate over the members of the ensemble
        for run in range(multi_runs):  # Each member will perform 'multi_runs' inferences
            preds = np.asarray(get_distributions_params(model=model_list[member], x=data_norm))
            all_preds.append(preds)
    all_preds = np.stack(all_preds, axis=0)

    # convert std to variance
    all_preds[:, :, 1] = np.square(all_preds[:, :, 1])

    return all_preds

def mul_predict(data_all_norm, input_features, loaded_models, means, stds, target="power_smoothed"):
    """ Computes predictions for each subset and denormalize them. Needs means and stds of input features.
    Returns a dictionnary with those predictions, keys are the subsets names. """
    predictions = {}
    for k in data_all_norm.keys():
        predictions[k] = {}
        inputs = torch.tensor(data_all_norm[k][input_features].values).float()
        preds_norm = get_ensemble_predictions(model_list=loaded_models,
                                            data_norm=inputs,
                                            multi_runs=10)
        predictions[k]["norm"] = preds_norm
        # Denormalize predicted mean
        preds_denorm = denorm_prediction(preds_norm=preds_norm,
                                        target_mean=means[target],
                                        target_std=stds[target])
        predictions[k]["denorm"] = preds_denorm
    return predictions



