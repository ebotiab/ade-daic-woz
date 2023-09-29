# -*- coding: utf-8 -*-
import click
import logging
import os

import numpy as np
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.daic_woz_datasets import DaicWozDataset
from src.features.build_processed_transcripts import add_cum_sample_vars
from src.hyperparameters import PARAMS
from src.models.train_model import evaluate_one_epoch, load_model, WeightedFocalLoss, filter_by_sex


def compute_predictions(test_dataset, model, device):
    """
    test model by creating a dataloader for each participant in the test dataset
    """
    criterion = WeightedFocalLoss(device) if PARAMS["use_focal_loss"] else nn.BCELoss()  # load loss
    metadata = test_dataset.metadata.copy()
    p_pred_dep_samples, p_logits = [], []
    for i, p_id in enumerate(metadata.index):  # test model for each participant
        # update dataset to only contain samples from one participant
        test_dataset.metadata = add_cum_sample_vars(metadata.loc[[p_id]])
        test_loader = DataLoader(test_dataset, PARAMS["batch_size"])
        # compute predictions for participant samples
        logits_i, labels_i = evaluate_one_epoch(model, test_loader, {}, criterion, 1e10, device, test_metrics=True)
        p_pred_dep_samples.append(logits_i.round().sum().item())  # save num of samples predicted as depressed
        p_logits.append(logits_i.mean().item())  # save mean of computed logits as the participant pred logit
    test_dataset.metadata = metadata
    test_dataset.metadata["n_pred_dep"] = p_pred_dep_samples
    test_dataset.metadata["n_pred_norm"] = test_dataset.metadata["num_samples"] - np.array(p_pred_dep_samples)
    test_dataset.metadata["logits"] = p_logits
    test_dataset.metadata["predictions"] = test_dataset.metadata["logits"].round()


def data_pred_to_metrics(data_pred, pred_norm_varname, pred_dep_varname):
    """
    compute metrics from metadata dataframe in which variables about sample and participant preds have been included
    """
    true_neg, false_neg = [data_pred.loc[data_pred["PHQ8_Binary"] == i, pred_norm_varname].sum() for i in [0, 1]]
    false_pos, true_pos = [data_pred.loc[data_pred["PHQ8_Binary"] == i, pred_dep_varname].sum() for i in [0, 1]]
    metrics = {"dep_pred_perc": (true_pos + false_pos) / (true_pos + false_pos + false_neg + true_neg) * 100,
               "accuracy": (true_pos + true_neg) / (true_pos + false_pos + false_neg + true_neg) * 100,
               "precision": true_pos / (true_pos + false_pos) * 100,
               "recall": true_pos / (true_pos + false_neg) * 100}
    metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    return metrics


def compute_test_metrics(metadata_pred, message=""):
    """
    compute sample and participant metrics from the computed predictions
    """
    metadata_pred["inverse_predictions"] = (~metadata_pred["predictions"].astype(bool)).astype(int)
    sample_metrics = data_pred_to_metrics(metadata_pred, "n_pred_norm", "n_pred_dep")
    participant_metrics = data_pred_to_metrics(metadata_pred, "inverse_predictions", "predictions")
    print(f"\n\n{message}\nSample metrics:\n{sample_metrics}\nParticipant metrics: {participant_metrics}")
    return sample_metrics, participant_metrics


def test_model(test_dataset, model, device, message=""):
    compute_predictions(test_dataset, model, device)
    sample_metrics, p_metrics = compute_test_metrics(test_dataset.metadata, message)
    return sample_metrics, p_metrics


@click.command()
@click.option("-p", '--data-path', type=click.Path(exists=True), default="data",
              help="folder path where data is located")
@click.option("-m", '--model-path', type=click.Path(exists=True), default="models/focal201_0_5_trval",
              help="path where the model to be evaluated is located")
@click.option('-e', '--model-epoch', type=int, default=-1,
              help="epoch from which is desired to load the trained model (if -1 load last available epoch)")
@click.option('-d', '--user_device', envvar="USER_DEVICE", prompt='User Device',
              help="device chosen to train the model, it must be one of 'cuda' or 'cpu'")
def main(data_path, model_path, model_epoch, user_device):  # TODO: fix options
    # load device
    device = user_device if torch.cuda.is_available() else "cpu"
    if device != user_device:  # load device
        print(f"'cpu' as device has been set since '{user_device}' device is not available")

    # load parameters
    spec_norm_type, r_seed = PARAMS["norm_spec_locally"], PARAMS["random_seed"]
    daic_woz_params = {"normalize_locally": spec_norm_type, "random_seed": r_seed, "data_path": data_path}

    for model_path in sorted(Path("models").iterdir()):
        model_path = str(model_path)

        # load back trained model
        print("-------------------------------")
        print("\n", model_path.upper(), "\n")

        cnn = load_model(device, model_path, model_epoch)

        # evaluate model in test dataset
        dwd_test = DaicWozDataset(f"{data_path}/processed/labels/test_metadata.csv", **daic_woz_params)
        test_model(dwd_test, cnn, device, "Model testing in entire dataset")

        # evaluate model in test separated by gender
        dwd_test_women = DaicWozDataset(f"{data_path}/processed/labels/test_metadata.csv", **daic_woz_params)
        filter_by_sex(dwd_test_women, "female")
        test_model(dwd_test_women, cnn, device, "Model testing in women")
        dwd_test_men = DaicWozDataset(f"{data_path}/processed/labels/test_metadata.csv", **daic_woz_params)
        filter_by_sex(dwd_test_men, "men")
        test_model(dwd_test_men, cnn, device, "Model testing in men")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
