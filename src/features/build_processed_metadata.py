# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.constants import CONSTANTS
from src.features import build_processed_transcripts

"""
Check possible errors in metadata and create processed metadata dataframes for each partition. It also contains the 
balance techniques that can be applied in the training set if selected in the hyperparameters configuration.
These dataframes will be used in EDA in streamlit app and in the torch dataset to be passed to the implemented models.
"""


@click.command()
@click.option("-p", '--data-path', type=click.Path(exists=True), default="data",
              help="folder path where data is located")
def main(data_path):
    # rename full_test_split file
    original_test_path = Path(data_path, "raw", "full_test_split.csv")
    if original_test_path.is_file():
        original_test_path.rename("test_split_Depression_AVEC2017.csv")

    # create metadata dfs
    partition_metadata_dfs = []
    for partition_name in ["train", "dev", "test"]:
        # load partition dataframe
        partition_path = str(Path(data_path, "raw", 'labels', f'{partition_name}_split_Depression_AVEC2017.csv'))
        partition_df = pd.read_csv(partition_path, index_col="Participant_ID")
        partition_df = partition_df.rename(columns={"PHQ_Binary": "PHQ8_Binary", "PHQ_Score": "PHQ8_Score"})

        # fix technical errors in metadata
        # check not null values in question variables
        if partition_name != "test":  # test set have not question score variables
            questions_df = partition_df.drop(["PHQ8_Binary", "PHQ8_Score", "Gender"], axis=1)
            # fill values in question variables for train metadata
            if partition_name == "train":
                questions_df = partition_df.drop(["PHQ8_Binary", "PHQ8_Score", "Gender"], axis=1)
                # TODO: change to be the logical
                max_grade, min_grade = CONSTANTS["max_question_grade"], CONSTANTS["min_question_grade"]
                mean_grade = np.ceil((max_grade - min_grade) / 2)  # (mean is rounded up to prioritize dep detection)
                partition_df[questions_df.columns] = questions_df.fillna(mean_grade)  # since one null value was found
                # since a binary label error was found in participant 409
                partition_df["PHQ8_Binary"] = (partition_df["PHQ8_Score"] >= 10).astype(int)

            # check label score correctness
            assert (partition_df[questions_df.columns].sum(axis=1) == partition_df["PHQ8_Score"]).all()

        # check not null values in partition dataframe
        assert partition_df.isna().sum().sum() == 0
        # check not errors in binary labels
        assert ((partition_df["PHQ8_Score"] >= 10) != partition_df["PHQ8_Binary"]).sum() == 0

        # keep only variables needed in the processed dataframes
        partition_df = partition_df[["PHQ8_Binary", "PHQ8_Score", "Gender"]]

        # add variables with the number of samples that can be extracted for each interview
        num_frames = np.empty(partition_df.shape[0])
        for i, participant_id in enumerate(partition_df.index):
            transcript = pd.read_csv(Path(data_path, "processed/transcripts", f"{participant_id}_TRANSCRIPT.csv"))
            num_frames[i] = transcript["last_sample"].iloc[-1]
        partition_df["num_samples"] = num_frames
        partition_df = build_processed_transcripts.add_cum_sample_vars(partition_df)

        # save partition metadata as csv file
        partition_df.to_csv(Path(data_path, "interim/labels", f"{partition_name}_metadata.csv"))  # for eda
        partition_df.to_csv(Path(data_path, "processed/labels", f"{partition_name}_metadata.csv"))  # for model

        # append to list with partition dfs to create global metadata
        partition_df["partition"] = partition_name
        partition_metadata_dfs.append(partition_df)

    # concat train and dev partition dfs and save in one single file (to train final models)
    train_dev_df = build_processed_transcripts.add_cum_sample_vars(pd.concat(partition_metadata_dfs[:2], axis=0))
    train_dev_df.to_csv(Path(data_path, "interim/labels", "train_dev_metadata.csv"))  # for eda
    train_dev_df.to_csv(Path(data_path, "processed/labels", "train_dev_metadata.csv"))  # for model

    # concat partition dfs and save in one single file (for EDA purposes)
    metadata_df = pd.concat(partition_metadata_dfs, axis=0)
    metadata_df = metadata_df.drop(["last_sample", "first_sample"], axis=1)
    metadata_df.to_csv(Path(data_path, "interim/labels", "metadata.csv"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    main()
