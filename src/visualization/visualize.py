# -*- coding: utf-8 -*-
import click
import os
from dotenv import find_dotenv, load_dotenv
import logging
from pathlib import Path
import plotly.express as px
from collections import OrderedDict
from copy import deepcopy
import pandas as pd
import torch
from torch import nn

from src.models import train_model
from src.hyperparameters import PARAMS


@click.command()
@click.option("-m", '--models-path', type=click.Path(exists=True), default="models",
              help="folder path where model is located")
@click.option("-p", '--data-path', type=click.Path(exists=True), default="data",
              help="folder path where data is located")
@click.option('-d', '--user_device', envvar="USER_DEVICE", prompt='User Device',
              help="device chosen to train the model, it must be one of 'cuda' or 'cpu'")
def main(models_path, data_path, user_device):
    metrics_df = pd.read_csv(Path(models_path, "metrics.csv"), index_col=0)
    """
    # load device
    device = user_device if torch.cuda.is_available() else "cpu"
    if device != user_device:
        print(f"'cpu' as device has been set since '{user_device}' device is not available")

    # TODO: check that the input model exists and facilitate its selection
    # models selection

    selected_model_names = [input("Enter the name of the model you want to visualize its training: "), ]
    first_model_path = Path(models_path, selected_model_names[0])
    model_last_epoch = int(str(list(sorted(first_model_path.iterdir()))[-1])[-6:-4])
    while input("Do you want to add another model? (y/n): ").lower() == "y":
        model_selected = input("Enter model name (its number of epochs must be greater or equal than first one): ")
        selected_model_names.append(model_selected)

    selected_model_names, model_last_epoch = ["exp2", "exp3", "exp4"], 4  # TODO
    criterion = train_model.WeightedFocalLoss(device) if PARAMS["use_focal_loss"] else nn.BCELoss()  # load loss

    # load datasets
    dwd_train, dwd_val = train_model.load_datasets(data_path)
    # create dataloaders
    train_dl, val_dl = train_model.create_dataloaders(dwd_train, dwd_val, PARAMS["batch_size"])

    # load epoch metrics
    metr = OrderedDict({"loss": [], "accuracy": [], "f1_score": [], "best_f1": [], "av_pr_score": [], "roc_auc": []})

    for model_name in selected_model_names:  # iterate through the epochs of each of the selected models
        model_path = Path(models_path, model_name)
        for epoch in range(model_last_epoch+1):
            # TODO: change to load saved metrics in train_model script

            training_metrics = deepcopy(metr)
            eval_metrics = deepcopy(training_metrics)
            # evaluate training
            
            train_model.evaluate_one_epoch(model, train_dl, training_metrics, criterion, PARAMS["max_batches"], device)

            model = train_model.load_model(device, model_path, epoch)
            eval_metrics = deepcopy(metr)
            # evaluate validation
            train_model.evaluate_one_epoch(model, val_dl, eval_metrics, criterion, PARAMS["max_batches"], device)
            # save loaded metrics
            # training_metrics["model"], training_metrics["epoch"] = model_name, epoch
            # training_metrics["data"] = "train_dev" if PARAMS["concat_train_dev"] else "train"
            eval_metrics["data"] = "dev" if PARAMS["concat_train_dev"] else "test"
            eval_metrics["model"], eval_metrics["epoch"] = model_name, epoch
            metrics_df = pd.concat([metrics_df, pd.DataFrame(eval_metrics)])
        metrics_df.to_csv(Path(models_path, "metrics.csv"))
    """

    # visualize loaded epoch metrics
    metrics_df = metrics_df.replace(
        ["exp0", "exp1", "exp2", "exp3", "exp4"],
        ["cbal=0.00", "cbal=0.5", "cbal=0.75", "cbal=0.25", "cbal=1.00"])
    metrics_df = metrics_df.sort_values(by=['model', "epoch"])
    for metric in metrics_df.drop(["model", "epoch", "data"], axis=1):
        fig = px.line(metrics_df, x="epoch", y=metric, color='model')
        fig.show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
