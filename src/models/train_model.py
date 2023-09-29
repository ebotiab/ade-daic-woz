# -*- coding: utf-8 -*-
import click
import logging
import os
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from copy import deepcopy
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import F1Score
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import mlflow

from src.features.build_processed_transcripts import add_cum_sample_vars
from src.models.cnn1d import CNN
from src.models import daic_woz_datasets
from src.hyperparameters import PARAMS

"""
To train model from processed data.
Run different experiments changing the hyper-parameters in the corresponding script.
"""


class WeightedFocalLoss(nn.Module):
    """
    Non weighted version of Focal Loss (taken from: https://amaarora.github.io/2020/06/29/FocalLoss.html)
    TODO: include alpha and gamma as training hyper-parameters
    """
    def __init__(self, device, alpha=.001, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss()(inputs, targets)  # modification from original
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-bce_loss)
        f_loss = at*(1-pt)**self.gamma * bce_loss
        return f_loss.mean()


# TODO: use the library that orders in semantic way the numbers
# TODO: check that if loaded from one epoch the epochs are stored with a filename from that number
def load_model(device, model_folder_path, model_epoch=5):
    """
    load specified model if exists, if not create from scratch. If model_epoch=-1 load from the last available epoch
    """
    cnn = CNN().to(device)  # create new model (without pretrained weights)
    models_folder = Path(model_folder_path)
    if models_folder.exists():  # load model from a trained version if the model folder exists
        model_path = str(list(sorted(models_folder.iterdir()))[model_epoch])
        print(model_path)
        state_dict = torch.load(model_path)
        cnn.load_state_dict(state_dict)
    # ensure that folder where checkpoints will be saved exists
    models_folder.mkdir(exist_ok=True)
    return cnn


def load_datasets(data_path):
    """
    load daic woz train and val datasets
    """
    # load dataset parameters
    spec_norm_type, r_seed = PARAMS["norm_spec_locally"], PARAMS["random_seed"]
    data_p = {"normalize_locally": spec_norm_type, "random_seed": r_seed, "data_path": data_path}

    # load train and val dataset paths
    if PARAMS["concat_train_dev"]:  # use train and dev as training set and test as val set
        train_path = str(Path(data_path, "processed/labels/train_dev_metadata.csv"))
        val_path = str(Path(data_path, "processed/labels/test_metadata.csv"))
    else:  # use train as training set and dev as validation set
        train_path = str(Path(data_path, "processed/labels/train_metadata.csv"))
        val_path = str(Path(data_path, "processed/labels/dev_metadata.csv"))

    # load daic-woz datasets
    if PARAMS["partition_id_level"] != 0:  # load subset of the dataset
        dwd_train = daic_woz_datasets.DaicWozIdSubset(train_path, False, PARAMS["partition_id_level"], **data_p)
        dwd_val = daic_woz_datasets.DaicWozIdSubset(train_path, True, PARAMS["partition_id_level"], **data_p)
    else:  # load entire datasets
        dwd_train = daic_woz_datasets.DaicWozDataset(train_path, **data_p)
        dwd_val = daic_woz_datasets.DaicWozDataset(val_path, **data_p)
    print("\nSize of the train dataset:", len(dwd_train), "\nSize of the validation dataset:", len(dwd_val))

    return dwd_train, dwd_val


def change_dep_threshold(dataset, new_threshold):
    metadata = dataset.metadata.copy()
    metadata["PHQ8_Binary"] = (metadata["PHQ8_Score"] >= new_threshold).astype(int)
    metadata_modified = metadata.loc[metadata["PHQ8_Binary"] != dataset.metadata["PHQ8_Binary"]]
    print("\nNew depression threshold in training set:", new_threshold)
    print("Number of interviews in which the class has been modified:", metadata_modified.shape[0])
    print("Number of samples in which the class has been modified:", metadata_modified["num_samples"].sum())
    dataset.metadata = metadata
    return metadata_modified


def filter_by_sex(dwd, sex_to_filter):
    metadata = dwd.metadata.copy()
    metadata = metadata.loc[metadata["Gender"] == int(sex_to_filter == "male")]
    metadata = add_cum_sample_vars(metadata)
    dwd.metadata = metadata


def create_dataloaders(dwd_train, dwd_val, batch_size, balance_type=None):
    weighted_sampler_train = create_weighted_sampler(dwd_train, balance_type) if balance_type else None
    train_dataloader = DataLoader(dwd_train, batch_size, shuffle=not balance_type, sampler=weighted_sampler_train)
    # change shuffle parameter depending on if all val used TODO: it is not the same?
    val_dataloader = DataLoader(dwd_val, batch_size, shuffle=True)
    return train_dataloader, val_dataloader


def create_weighted_sampler(dataset, balance_type):
    """
    balance global ids if undersampling, create weighted sampler with classes and ids of each class balanced otherwise
    """
    metadata, n_samples = dataset.metadata, len(dataset)
    n_depressed = int(metadata.loc[metadata["PHQ8_Binary"] == 1, "num_samples"].sum())
    n_not_depressed = int(metadata.loc[metadata["PHQ8_Binary"] == 0, "num_samples"].sum())
    weights = np.empty(n_samples)
    assert balance_type in ["classes", "classes_ids", "ids"]
    for interview_id, interview_data in metadata.iterrows():
        first_sample_i, last_sample_i = int(interview_data["first_sample"]), int(interview_data["last_sample"])
        n_samples_i = metadata.loc[interview_id, "num_samples"]
        n_label_samples = n_depressed if interview_data["PHQ8_Binary"] else n_not_depressed
        if balance_type == "classes":  # create weights to balance classes
            weights[first_sample_i:last_sample_i] = 1 / n_label_samples
        elif balance_type == "classes_ids":  # create weights to balance ids for each class and classes
            weights[first_sample_i:last_sample_i] = n_samples / (n_samples_i * n_label_samples)
        elif balance_type == "ids":  # create weights to balance global ids
            weights[first_sample_i:last_sample_i] = n_samples / n_samples_i
    weights = torch.from_numpy(weights).type('torch.DoubleTensor')
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler


def reduce_learning_rate(optimizer):
    """
    Reduce the learning rate of the optimiser for training

    Input
        optimiser: obj - The optimiser setup at the start of the experiment
    """
    for param_group in optimizer.param_groups:
        print('Reducing Learning rate from: ', param_group['lr'],
              ' to ', param_group['lr'] * PARAMS["lr_decay_factor"])
        param_group['lr'] *= PARAMS["lr_decay_factor"]


def train(model, train_loader, val_loader, criterion, optimizer, n_epochs, max_batches, device, model_path):
    # create empty dictionaries to store metrics
    metrics = OrderedDict({"losses": [], "accuracies": [], "f1_scores": [], "best_f1": [], "p_auc": [], "r_auc": []})
    tr_metrics, ev_val_met, ev_tr_met = deepcopy(metrics), deepcopy(metrics), deepcopy(metrics)
    for i in range(n_epochs):
        # train and evaluate in each epoch
        print(f"Epoch {i + 1}")
        time_start_epoch = time.time()
        print("Train:"), train_one_epoch(model, train_loader, criterion, optimizer, tr_metrics, max_batches, device)
        print("Val:"), evaluate_one_epoch(model, val_loader, ev_val_met, criterion, max_batches, device)
        if PARAMS["evaluate_training"]:
            print("Val train"), evaluate_one_epoch(model, train_loader, ev_tr_met, criterion, max_batches, device)
        print("\nEpoch time duration:", time.time() - time_start_epoch, "seconds\n")

        # apply learning rate decay
        if (i+1) % PARAMS["lr_decay_freq"] == 0 and PARAMS["lr_decay_freq"] != -1:
            reduce_learning_rate(optimizer)

        # display metrics
        if (i+1) % PARAMS["plot_metrics_frequency"] == 0:
            for metrics_type in metrics:
                plot_results([tr_metrics, ev_val_met, ev_tr_met], metrics_type, model_path)

        # save checkpoint (model state for current epoch)
        model_file_path = Path(model_path, f"epoch_{i}.pth") if i > 9 else Path(model_path, f"epoch_0{i}.pth")
        torch.save(model.state_dict(), model_file_path)

    print("\n\nFinished training")


def train_one_epoch(model, train_loader, criterion, optimizer, train_metrics, max_batches, device):
    # initialize metrics
    all_logits, all_labels = [], []

    for batch_counter, (specs, labels, idxs) in enumerate(tqdm(train_loader)):

        # training loop
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()  # reset gradients
        logits = model(specs).view(-1)  # get batch predictions
        loss_tensor = criterion(logits, labels.float())  # get batch loss
        loss_tensor.backward()  # back-propagate error
        optimizer.step()  # update weights

        # save labels and logits
        all_labels += labels.tolist()
        all_logits += logits.tolist()

        # stop epoch when limit of batches is reached
        if batch_counter == max_batches:
            break

    # compute epoch metrics
    all_logits = torch.tensor(all_logits).to(device)
    all_labels = torch.tensor(all_labels).to(device).type(torch.int64)
    save_epoch_metrics(train_metrics, all_logits, all_labels, criterion, device)


def evaluate_one_epoch(model, eval_loader, eval_metrics, criterion, max_batches, device, test_metrics=False):
    with torch.no_grad():  # disable gradients
        all_logits, all_labels = [], []

        model.eval()  # disable training mode
        for batch_counter, (specs, labels, idxs) in enumerate(tqdm(eval_loader)):
            # compute predictions
            specs = specs.to(device)
            logits = model(specs).view(-1)
            # save labels and predictions
            all_logits += logits.tolist()
            all_labels += labels.tolist()

            # stop epoch when limit of batches is reached
            if batch_counter == max_batches:
                break
        model.train()  # enable training mode

        # compute epoch metrics
        all_logits = torch.tensor(all_logits).to(device)
        all_labels = torch.tensor(all_labels).to(device).type(torch.int64)
        if not test_metrics:
            save_epoch_metrics(eval_metrics, all_logits, all_labels, criterion, device)
    return all_logits, all_labels


def save_epoch_metrics(metrics, logits, labels, criterion, device):
    """
    compute, save and display epoch metrics from given logits
    """
    # TODO: save metrics as a row of ths csv that should be created not store them in pkl, and hyperparameters as pickle
    epoch_metrics = compute_metrics(logits, labels, criterion, device)
    # TODO: mlflow.log_metrics(metrics) if  # save metrics in mlflow
    [metrics[m1].append(epoch_metrics[m2]) for m1, m2 in zip(metrics, epoch_metrics)]  # save epoch metrics
    [print(f"{m}: {metrics[m][-1]}") for m in metrics]  # display epoch metrics


def compute_metrics(logits, labels, criterion, device):
    """
    compute epoch metrics from given logits
    """
    metrics = OrderedDict({})
    metrics["loss"] = criterion(logits, labels.float()).item()
    metrics["accuracy"] = get_accuracy(logits, labels)
    metrics["f1_score"] = F1Score(num_classes=1).to(device)(logits, labels).item()
    precision, recall, thresholds = precision_recall_curve(labels.float().cpu(), logits.cpu())
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    metrics["best_f1"] = np.max(f1_scores)
    metrics["p_auc"] = average_precision_score(labels.float().cpu(), logits.cpu())
    metrics["r_auc"] = roc_auc_score(labels.float().cpu(), logits.cpu())
    return metrics


def get_accuracy(probabilities, labels):
    predictions = (probabilities > 0.5).int()
    are_correct = predictions == labels.view(*predictions.shape)
    return torch.mean(are_correct.type(torch.FloatTensor)).item() * 100


def plot_results(all_metrics, metrics_type, model_folder):
    tr_results, eval_val_results, eval_tr_results = [m[metrics_type] for m in all_metrics]
    plot_title = f'{PARAMS["model_name"]}_{metrics_type}'
    plt.title(plot_title)
    plt.plot(tr_results, label="training")
    plt.plot(eval_val_results, label="eval_validation")
    _ = plt.plot(eval_tr_results, label="eval_training") if PARAMS["evaluate_training"] else None
    plt.legend(frameon=False)
    # save results
    plt.savefig(Path(model_folder, f'{plot_title}.png'))
    with open(Path(model_folder, f'{plot_title}_train.pkl'), "wb") as f:
        pickle.dump(tr_results, f)
    with open(Path(model_folder, f'{plot_title}_eval.pkl'), "wb") as f:
        pickle.dump(eval_val_results, f)
    plt.show()


@click.command()
@click.option("-p", '--data-path', type=click.Path(exists=True), default="data",
              help="folder path where data is located")
@click.option('-d', '--user_device', envvar="USER_DEVICE", prompt='User Device',
              help="device chosen to train the model, it must be one of 'cuda' or 'cpu'")
@click.option("-m", '--models-path', type=click.Path(exists=True), default="models",
              help="folder path where models are located")
def main(data_path, user_device, models_path):
    # display selected parameters for the experiment
    print(f"SELECTED PARAMS IN {PARAMS['model_name']} EXPERIMENT:")
    print(f"Bath Size: {PARAMS['batch_size']}\nLearning rate: {PARAMS['lr_rate']}\nNº batches: {PARAMS['max_batches']}")

    # load device
    device = user_device if torch.cuda.is_available() else "cpu"
    if device != user_device:
        print(f"'cpu' as device has been set since '{user_device}' device is not available")

    # create model or load from trained one if desired
    model_path = Path(models_path, PARAMS["model_name"])
    cnn = load_model(device, model_path)

    # load loss function  # TODO: research about weight losses
    criterion = WeightedFocalLoss(device) if PARAMS["use_focal_loss"] else nn.BCELoss()

    # load datasets
    dwd_train, dwd_val = load_datasets(data_path)

    # filter by sex if specified
    if PARAMS["filter_by_sex"]:
        filter_by_sex(dwd_train, PARAMS["filter_by_sex"])
        filter_by_sex(dwd_val, PARAMS["filter_by_sex"])

    # change the depression threshold in training set if specified
    if PARAMS["dep_train_threshold"] != 10:
        change_dep_threshold(dwd_train, PARAMS["dep_train_threshold"])

    # create dataloaders
    train_dl, val_dl = create_dataloaders(dwd_train, dwd_val, PARAMS["batch_size"], PARAMS["weights_balance_type"])

    # load optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=PARAMS["lr_rate"])

    # save parameters in mlflow
    mlflow.set_experiment(experiment_name=PARAMS["model_name"])
    mlflow.log_params(PARAMS)

    # train model
    n_epochs, max_batches, model_p = PARAMS["n_epochs"],  PARAMS["max_batches"], Path(models_path, PARAMS['model_name'])
    train(cnn, train_dl, val_dl, criterion, optimizer, n_epochs, max_batches, device, model_p)

    # save model
    model_folder_path = Path(models_path, PARAMS['model_name'])
    print(f"Trained model saved at {model_folder_path}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
