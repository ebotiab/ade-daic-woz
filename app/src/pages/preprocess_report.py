import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.src import constants
from src.features import build_processed_transcripts
from src.models.daic_woz_datasets import DaicWozDataset
from src.features.utils_audio import display_interactive_audio
from src.hyperparameters import PARAMS


def include_all_samples_plot(sample_hop, audio_time, plot_start, plot_end, sample_highlight):
    sample_time = PARAMS["sample_time"]
    x_coords = [(i, i+sample_time) for i in np.arange(0, audio_time-sample_time, sample_hop)]
    for i, xs in enumerate(x_coords):
        _ = plt.axvline(x=xs[0], ymin=0.05, ymax=0.95) if plot_start else None
        _ = plt.axvline(x=xs[1], ymin=0.05, ymax=0.95) if plot_end else None
        if i == sample_highlight:
            plt.axvline(x=xs[0], ymin=0.05, ymax=0.95, color="r")
            plot = plt.axvline(x=xs[1], ymin=0.05, ymax=0.95, color="r")
            y_min, y_max = plot.axes.get_ylim()
            plt.fill_betweenx(y=[y_min, y_max], x1=xs[0], x2=xs[1], alpha=0.3)
        _ = plt.axvline(x=xs[0], ymin=0.05, ymax=0.95, color="r") if i == sample_highlight+sample_time/sample_hop else _


def segment_extraction_report():
    st.write(constants.segment_extraction_1)
    transcript_345 = pd.read_csv("data/raw/transcripts/345_TRANSCRIPT.csv", sep="\t")
    transcript_345["speaker"] = transcript_345["speaker"].replace("Interview", "Participant")
    st.write("0. Transcript from interview 345:", transcript_345.head())
    transcript_345 = build_processed_transcripts.remove_support_comments(transcript_345)  # remove support comments
    st.write("1. Removal of overlapping comments", transcript_345.head())
    transcript_345 = build_processed_transcripts.group_speaking(transcript_345)  # create transcript with speaking times
    st.write("2. Speaker aggregation:", transcript_345.head(6))
    transcript_345 = transcript_345.loc[transcript_345["speaker"] == "Participant"].reset_index(drop=True)  # no ellie
    transcript_345 = build_processed_transcripts.add_num_samples(transcript_345, PARAMS["sample_hop"])  # add n samples
    st.write("3. Removal of Ellie instances:", transcript_345.drop("num_samples", axis=1).head())
    st.write("4. Extraction of number of samples:", transcript_345.head())
    transcript_345 = transcript_345.loc[transcript_345["num_samples"] != 0]  # filter segments without samples
    st.write("5. Removal of short fragments:", transcript_345.head())
    col_types = {"last_sample": int, "first_sample": int}
    processed_transcript_file = "data/processed/transcripts/345_TRANSCRIPT.csv"
    processed_transcript = pd.read_csv(processed_transcript_file, dtype=col_types).drop(["speaker", "value"], axis=1)
    st.write("6. Final transcript dataframe:", processed_transcript)


def app():
    st.write(constants.preprocess_report_intro)

    # segment extraction section
    st.write("### 1. Segment extraction")
    if st.checkbox("Display Segment extraction"):
        segment_extraction_report()

    # number of samples extraction section
    st.write(constants.n_sample_extraction_1)
    dwd = DaicWozDataset("data/processed/labels/train_metadata.csv")
    idx_chosen = 1
    segment = dwd.get_segment(idx_chosen)
    st.write(f"Consider, for example, the segment {segment.id} from the interview {segment.participant_id},",
             f"which starts in {segment.audio_times[0]} and ends in {segment.audio_times[1]}.",
             "Its audio waveform and spectrogram would be:")

    # display segmented plots
    plt.plot()
    waveform_plot, audio, sr = segment.show_waveform()
    display_interactive_audio(audio, sr, is_tensor=True)  # display interactive audio widget
    segment_time = segment.audio_times[1] - segment.audio_times[0]
    c1, c2, c3 = st.columns((.25, .375, .375))
    plot_start, plot_end = c1.checkbox("Include start sample times"), c1.checkbox("Include end sample times")
    sample_hop = c2.number_input("Select sample hop (in seconds)", value=PARAMS["sample_hop_time"])
    n_samples = int(np.ceil((segment_time - PARAMS["sample_time"]) / sample_hop))
    sample_highlight = c3.number_input("Select sample to highlight", max_value=n_samples-1, value=n_samples-1, step=1)
    include_all_samples_plot(sample_hop, segment_time, plot_start, plot_end, sample_highlight)
    waveform_plot.axes[0].get_xaxis().set_visible(False)
    st.pyplot(plt)  # display segmented waveform
    spec_plot, spec = segment.show_spectrogram(add_color_bar=False)
    include_all_samples_plot(sample_hop, segment_time, plot_start, plot_end, sample_highlight)
    st.pyplot(spec_plot)  # display segmented spec
    st.write(constants.n_sample_extraction_2)
    st.write("From this segment we can obtain a total", n_samples, "samples")  # write conclusions
    # st.write("Therefore, the following variables will be added to the transcript dataframe of this interview")
    # st.write(pd.read_csv("data/processed/transcripts/303_TRANSCRIPT.csv").head())
    # librosa.time_to_frames(sample_hop, sr=PARAMS["sr"], **PARAMS["stft"]) - PARAMS["first_frame_ind"]
    # n_samples_f = int(np.ceil((spec.shape[1] - PARAMS["sample_n_frames"])/sample_hop_frames))
    # assert n_samples == n_samples_f # must result in the same number of samples



