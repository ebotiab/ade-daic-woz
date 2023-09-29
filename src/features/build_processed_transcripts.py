# -*- coding: utf-8 -*-
import click
import logging
import os
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import numpy as np
import pandas as pd

from src.hyperparameters import PARAMS


"""
To create transcript dataframes where each instance represents a interview segment.
Information about the number of samples that can be extracted is included for each segment instance.
See streamlit app 'preprocess_report' section for more clarification about the pre-processing pipeline.
"""


def remove_support_comments(transcript):
    """
    drop from transcript all instances that are between start and stop times of previous instances
    """
    support_comments = (transcript["start_time"] > transcript["start_time"].shift()) & \
                       (transcript["stop_time"] < transcript["stop_time"].shift())
    return transcript.loc[~support_comments]


def group_speaking(transcript, time_diff=None):
    """
    group instances where consecutive speaker does not change or, if time_diff is not None, where stop_time and
    start_times have difference lower than time_diff. Start time is taken from the first instance start and stop times
    from the last one. Transcript values are concatenated separating them with a dot (". ")
    """
    if time_diff:
        groups = (transcript["stop_time"] - transcript["start_time"].shift() > time_diff).cumsum()
    else:
        groups = (transcript["speaker"] != transcript["speaker"].shift()).cumsum()
    aggregations = {"start_time": "first", "stop_time": "last", "speaker": "first", "value": lambda x: ". ".join(x)}
    return transcript.groupby(groups, as_index=False).agg(aggregations)


def add_silence_times(transcript, use_post_silences=False):
    """
    add silences in segments using stop_time from prev segments and, if desired, start_time from post segments
    """
    transcript["start_time"] = np.minimum(transcript["stop_time"].shift(1, fill_value=0), transcript["start_time"])
    if use_post_silences:
        transcript["stop_time"] = np.maximum(transcript["stop_time"], transcript["start_time"].shift(-1, fill_value=0))
    return transcript


def add_num_samples(transcript, sample_hop):
    """
    add column with the number of spectrogram samples, which are build from PARAMS["sample_time"] audio seconds,
    that are possible to extract from the audio segments.
    """
    segment_time = transcript["stop_time"] - transcript["start_time"]
    transcript["num_samples"] = (((segment_time - PARAMS["sample_time"]) // sample_hop) + 1).astype(int)
    transcript.loc[transcript["num_samples"] < 0, "num_samples"] = 0  # segment times lower than PARAMS["sample_time"]
    return transcript


def correct_async_transcripts(interview_id, transcript):
    """
    synchronizes interview if it belongs to a predefined list.
    Out-of-sync interviews found thanks to https://github.com/adbailey1/daic_woz_process
    """
    if interview_id in [318, 321, 341, 362]:  # TODO: check them
        # compute time differences in async transcripts
        audio_starts = {318: 46.5, 321: 38, 341: 50.5, 362: 26}  # extracted by listening the corresponding audios
        real_start = audio_starts[interview_id]
        transcript_start = transcript.iloc[0]["start_time"].item()
        time_diff = real_start - transcript_start
        # sync transcript times
        transcript["start_time"] += time_diff
        transcript["stop_time"] += time_diff
    return transcript


def add_cum_sample_vars(transcript):
    transcript.loc[:, "last_sample"] = transcript["num_samples"].cumsum().astype(int)
    transcript.loc[:, "first_sample"] = transcript["last_sample"].shift(fill_value=0)
    return transcript


def process_transcript(transcript_files_path, transcript_file_name, interim_path, use_prev_silences):
    """
    pre-processing transcript pipeline
    """
    # load interview data
    interview_id = int(transcript_file_name[:3])
    transcript = pd.read_csv(Path(transcript_files_path, transcript_file_name), sep="\t")
    # pre-processing pipeline
    transcript = correct_async_transcripts(interview_id, transcript)  # fix async transcripts
    transcript["value"] = transcript["value"].fillna("")  # fill possible null values with the empty string
    if (transcript["speaker"].unique() == "Interview").all():  # for transcripts with only interview segments
        transcript = group_speaking(transcript, time_diff=TIME_SEP_SPEAKER)
    else:
        transcript = remove_support_comments(transcript)  # remove 'support comments'
        # create transcript where each instance belong to a speaking time (or interview segment)
        transcript = group_speaking(transcript)
        # include in start times from participant segments silence duration before speaking if desired
        transcript = add_silence_times(transcript) if use_prev_silences else transcript
        # filter out instances where participant is not speaking
        transcript = transcript.loc[transcript["speaker"] == "Interview"].reset_index(drop=True)
    transcript = add_num_samples(transcript, PARAMS["sample_hop_time"])  # add col with number of samples to extract
    # save transcript at this stage to eda analysis in the streamlit app
    transcript.to_csv(Path(interim_path, transcript_file_name), index=False)
    # filter segments where there are no samples that can be extracted
    transcript = transcript.loc[transcript["num_samples"] != 0]
    # reformat num_samples into cumulative variables 'first_sample' and 'last_sample' to facilitate sample search
    transcript = add_cum_sample_vars(transcript)
    return transcript


@click.command()
@click.option("-d", '--data-p', type=click.Path(exists=True), default="data",
              help="Path where data is located")
def main(data_p):
    """
    build processed transcripts
    """
    transcript_p, interim_p, output_p = [str(Path(data_p, i, "transcripts")) for i in ["raw", "interim", "processed"]]
    for f in os.listdir(transcript_p):
        if f.endswith("TRANSCRIPT.csv"):
            # get processed transcript
            transcript = process_transcript(transcript_p, f, interim_p, PARAMS["use_silences"])

            if transcript.shape[0] == 0:  # for transcripts where has not been possible to extract any segment
                # try the same process but taking previous silences in the segments
                transcript = process_transcript(transcript_p, f, interim_p, use_prev_silences=True)
                # raise an error if the problem persists
                assert transcript.shape[0] > 0

            # check that there are no null values in the transcript df
            assert transcript.isna().sum().sum() == 0

            # save processed transcript
            transcript.to_csv(Path(interim_p, f), index=False)  # save processed to interim folder to perform eda
            transcript.to_csv(Path(output_p, f), index=False)  # save into processed folder to pass the model


if __name__ == '__main__':
    # define hyperparameter constants
    TIME_SEP_SPEAKER = PARAMS["time_sep_speaker"]
    SAMPLE_TIME = PARAMS["sample_time"]
    STFT_PARAMS = PARAMS["stft"]

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
