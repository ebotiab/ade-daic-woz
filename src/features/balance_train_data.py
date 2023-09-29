# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

from src.hyperparameters import PARAMS
from src.features import build_processed_transcripts

"""
Balance training datasets depending on the hyperparameters configuration
"""


def get_class_metadata(metadata, is_dep):
    class_metadata = metadata.loc[metadata["PHQ8_Binary"] == is_dep].copy()
    class_metadata["cum_n_samples"] = class_metadata["num_samples"].cumsum().shift(fill_value=0)
    return class_metadata


def num_undersampling(max_n, balanced_n, undersampling_level):
    """
    compute the number of instances to keep according to undersampling_level
    """
    n_undersampling = max_n - undersampling_level * (max_n - balanced_n)
    return int(n_undersampling)


def reduction_changes_str(metadata_old, metadata_balanced, class_type_reduction, message=""):
    to_print = [message] if message else []
    if class_type_reduction:
        removed_interviews = metadata_old.drop(metadata_balanced.index)
        removed_ids, removed_num_samples = removed_interviews.index, removed_interviews["num_samples"].astype(int)
        to_print.append(f"Number of interviews in original dataset: {len(metadata_old)}")
        to_print.append(f"Number of interviews in new dataset: {len(metadata_balanced)}")
        to_print.append(f"Percentage of interviews removed: {round(len(removed_ids)/len(metadata_old)*100, 2)}%")
        to_print.append(f"Ids from removed interviews: {list(zip(removed_ids, removed_num_samples))}")
    old_num_samples, num_samples = [int(df['num_samples'].sum()) for df in [metadata_old, metadata_balanced]]
    to_print.append(f"Number of samples in original dataset: {old_num_samples}")
    to_print.append(f"Number of samples in balanced dataset: {num_samples}")
    perc_samples_removed = round((old_num_samples - num_samples)/old_num_samples*100, 2)
    to_print.append(f"Percentage of samples removed: {perc_samples_removed}%")
    return to_print


def class_undersampling(metadata, und_level):
    metadata_old = metadata.copy()
    metadata = metadata.sort_values("num_samples", ascending=not PARAMS["class_bal_max_data"])
    dep_met, norm_met = get_class_metadata(metadata, 1), get_class_metadata(metadata, 0)
    max_n, min_n = [agg(dep_met["num_samples"].sum(), norm_met["num_samples"].sum()) for agg in [max, min]]
    bal_n = num_undersampling(max_n, min_n, und_level)
    # select the interviews in the balance dataset
    dep_bal, norm_bal = dep_met.loc[dep_met["cum_n_samples"] < bal_n],  norm_met.loc[norm_met["cum_n_samples"] < bal_n]
    balanced_ids = np.concatenate([dep_bal.index, norm_bal.index])
    metadata = metadata.loc[balanced_ids]  # filter metadata by the selected ids
    # display undersampling changes in data
    [print(i) for i in reduction_changes_str(metadata_old, metadata, True, "\nClass balance undersampling:")]
    return build_processed_transcripts.add_cum_sample_vars(metadata)  # update 'last_sample' and 'first_sample' columns


def ids_undersampling(metadata, und_level):
    metadata_old = metadata.copy()
    num_samples = num_undersampling(metadata["num_samples"].max(), metadata["num_samples"].min(), und_level)
    metadata["num_samples"] = np.minimum(metadata["num_samples"], num_samples)  # ids undersampling
    # display undersampling changes in data
    [print(i) for i in reduction_changes_str(metadata_old, metadata, False, "\nIds balance undersampling:")]
    return build_processed_transcripts.add_cum_sample_vars(metadata)  # update 'last_sample' and 'first_sample' columns


def ids_few_samples_filter(metadata, filter_level):
    """
    remove interviews that have extreme low outliers in their number of samples.
    When filter_level=0.6667 only remove outliers, when =0.3333 only remove extreme outliers
    """
    metadata_old, num_samples = metadata.copy(), metadata["num_samples"]
    q1, iqr = num_samples.quantile(0.25), num_samples.quantile(0.75) - num_samples.quantile(0.25)
    metadata = metadata.loc[~(num_samples < (q1 - (1/(filter_level+1e-10))*iqr))]
    [print(i) for i in reduction_changes_str(metadata_old, metadata, True, f"\nIds few samples filter:")]
    return build_processed_transcripts.add_cum_sample_vars(metadata)  # update 'last_sample' and 'first_sample' columns


def get_sample_hop(num_samples, transcript):
    segment_time = transcript["stop_time"] - transcript["start_time"]
    return ((segment_time - PARAMS["sample_time"]) / num_samples).sum()


def get_num_samples(sample_hop, transcript):
    segment_time = transcript["stop_time"] - transcript["start_time"]
    num_samples = np.ceil((segment_time - PARAMS["sample_time"]) / sample_hop).astype(int)
    return num_samples.sum()


def get_precise_x(fun, approx_x, target_y, *kwargs):  # TODO: improve algorithm
    # only applicable if fun(approx_x)  is a lower than target_y
    niter = 0
    original_x = approx_x
    while fun(approx_x, *kwargs) != target_y:
        niter += 1
        approx_x += 0.001
        if niter > 100000:
            print(original_x)
    return approx_x


def load_hop_bal_n_samples(metadata, hop_balance_index, transcripts_path):  # TODO: result can be saved into a pickle
    """
    return number of sample to use in hop balance (more info in preprocessing report in the streamlit application)
    """
    min_sample_hop_time, num_samples = librosa.samples_to_time(PARAMS["stft"]["hop_length"], sr=PARAMS["sr"]), []
    for participant_id in metadata.index:
        transcript_path = str(Path(transcripts_path, f"{participant_id}_TRANSCRIPT.csv"))
        transcript = pd.read_csv(transcript_path)
        transcript = build_processed_transcripts.add_num_samples(transcript, min_sample_hop_time)
        num_samples.append(transcript["num_samples"].sum())
    return sorted(num_samples)[hop_balance_index]


def hop_balance(metadata, hop_balance_level, hop_balance_index, transcripts_path):
    """
    Perform hop sample balance (more info in preprocessing report in the streamlit application)
    """
    # get number of samples to use in hop balance
    min_bal_num_samples = load_hop_bal_n_samples(metadata, hop_balance_index, transcripts_path)
    metadata_old = metadata.copy()
    all_sample_hops, all_num_samples = [], []
    for participant_id in metadata.index:
        # load interview transcript data
        transcript_path = str(Path(transcripts_path, f"{participant_id}_TRANSCRIPT.csv"))
        transcript = pd.read_csv(transcript_path)
        # obtain new interview sample hop
        bal_num_samples = num_undersampling(transcript.iloc[-1]["last_sample"], min_bal_num_samples, hop_balance_level)
        sample_hop = get_sample_hop(bal_num_samples, transcript)
        # TODO: sample_hop = get_precise_x(get_num_samples, sample_hop, bal_num_samples, transcript)
        # update interview transcript data with new sample_hop
        transcript = build_processed_transcripts.add_num_samples(transcript, sample_hop)
        transcript = build_processed_transcripts.add_cum_sample_vars(transcript)
        transcript.to_csv(transcript_path, index=False)
        # save data with which metadata will be updated
        all_sample_hops.append(sample_hop)
        all_num_samples.append(transcript["last_sample"].iloc[-1])
    # add variable with sample hops data and update metadata with the new number of samples
    metadata["sample_hop"], metadata["num_samples"] = all_sample_hops, all_num_samples
    [print(i) for i in reduction_changes_str(metadata_old, metadata, False, "\nHop balance:")]
    return build_processed_transcripts.add_cum_sample_vars(metadata)  # update 'last_sample' and 'first_sample' columns


def balance_data(metadata, data_path, balance_params):
    # balance train ids with hop-balance
    if balance_params["hop_balance_level"] != 0:
        transcripts_path = str(Path(data_path, "processed/transcripts/"))
        hop_b_level, hop_b_index = balance_params["hop_balance_level"], balance_params["hop_balance_index"]
        metadata = hop_balance(metadata, hop_b_level, hop_b_index, transcripts_path)

    # balance train ids with under-sampling
    if balance_params["ids_undersampling_level"] != 0:
        metadata = ids_undersampling(metadata, balance_params["ids_undersampling_level"])

    # balance train classes with under-sampling
    if balance_params["class_undersampling_level"] != 0:
        metadata = class_undersampling(metadata, balance_params["class_undersampling_level"])

    # remove interview with extreme few samples
    if balance_params["ids_few_samples_filter_level"] != 0:
        metadata = ids_few_samples_filter(metadata, balance_params["ids_few_samples_filter_level"])

    return metadata


@click.command()
@click.option("-p", '--data-path', type=click.Path(exists=True), default="data",
              help="folder path where data is located")
def main(data_path):
    for train_partition_name in ["train", "train_dev"]:  # balance train and concatenation of val and training data
        train_partition_path = str(Path(data_path, f"processed/labels/{train_partition_name}_metadata.csv"))
        train_metadata = pd.read_csv(train_partition_path, index_col="Participant_ID")  # load data
        train_metadata = balance_data(train_metadata, data_path, PARAMS)  # balance train dataset
        train_metadata.to_csv(train_partition_path)  # save balanced metadata dataframe


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    main()
