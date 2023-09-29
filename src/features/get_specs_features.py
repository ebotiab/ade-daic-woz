# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
import pickle
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from src.models import daic_woz_datasets
import torch
from torch.utils.data import DataLoader

from src.constants import CONSTANTS
from src.hyperparameters import PARAMS

"""
Get aggregated spectrogram features from training dataset. They will be used to normalize the input model data.
"""
# TODO: it could be implemented a way to continue computing the features from stored ones (specifying their nº batches)


def set_spec_features(spec_features, spec_features_path):
    file = open(spec_features_path, "wb")
    pickle.dump(spec_features, file)
    file.close()


def load_spec_features(spec_features_path):
    if not Path(spec_features_path).is_file():  # save default if features they have not been extracted
        set_spec_features(CONSTANTS["init_spec_features"], spec_features_path)
    with open(spec_features_path, 'rb') as f:  # load spec features
        spec_features = pickle.load(f)
    return spec_features


@click.command()
@click.option("-m", '--train-metadata-path', type=click.Path(exists=True),
              default="data/processed/labels/train_metadata.csv", help="path to train metadata file")
@click.option('-d', '--user-device', envvar="USER_DEVICE", prompt='User Device',
              help="device chosen to perform torch operations, it must be one of 'cuda' or 'cpu'")
@click.option("-o", '--output-path', type=click.Path(exists=True), default="data/processed/",
              help="Output folder")
def main(train_metadata_path, user_device, output_path):
    # load train dataloader
    device = user_device if torch.cuda.is_available() else "cpu"
    dwd = daic_woz_datasets.DaicWozDataset(train_metadata_path, device=device, normalize_locally=False)
    batch_size = PARAMS["batch_size"]  # (value not relevant for this task)
    train_dataloader = DataLoader(dwd,  batch_size, shuffle=True)

    # create or set spec features to the initial ones
    output_file_path = Path(output_path, "spec_features.txt")
    set_spec_features(CONSTANTS["init_spec_features"], output_file_path)

    # start feature extraction process
    for batch_counter, (specs, _, _) in enumerate(tqdm(train_dataloader)):
        specs = specs.to(device)
        if batch_counter == 0:
            # get first batch features
            specs_mean, specs_var, specs_min, specs_max = specs.mean(), specs.var(), specs.min(), specs.max()
        elif batch_counter == PARAMS["get_features_max_batches"]:
            # stop extraction process if batch has reached the max number of batch
            break
        else:
            # update global spec features with extracted batch features
            cum_n_specs = batch_counter * batch_size
            total_specs = cum_n_specs + batch_size
            specs_mean = specs_mean*(cum_n_specs/total_specs) + specs.mean()*(batch_size/total_specs)
            specs_var = ((cum_n_specs-1)*specs_var + (batch_size-1)*(specs.var())) / (cum_n_specs+batch_size-1)
            specs_min, specs_max = min(specs_min, specs.min()), max(specs_max, specs.max())
            # print(specs_mean.item(), specs.mean().item(), specs_var.item(), specs.var().item())  # for debugging

    # save spec features into a file
    specs_mean, specs_var, specs_min, specs_max = [i.item() for i in [specs_mean, specs_var, specs_min, specs_max]]
    spec_features = {"global_mean": specs_mean, "global_var": specs_var, "min": specs_min, "max": specs_max}
    set_spec_features(spec_features, output_file_path)

    # load saved features and display them
    spec_features = load_spec_features(output_file_path)
    print("SPEC FEATURE EXTRACTED:", spec_features)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
