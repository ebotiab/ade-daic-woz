import pandas as pd
import streamlit as st
import random
import os
from datetime import timedelta
from pathlib import Path
import plotly.express as px
import torch
from torch import nn

from app.src.pages import global_analysis
from src.hyperparameters import PARAMS
from src.features.utils_audio import secs2time, display_interactive_audio
from src.models import daic_woz_datasets, custom_classes
from src.models import train_model

# TODO: Add integer input to select sample

# executing script mode (change if you want to debug the code), since st session state does not work when debugging
DEBUGGING = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# folder paths (change if the project folder structure is different)
METADATA_FILE_PATH = "data/processed/labels/individual_analysis.csv"
MODELS_PATH = "models"


def fragment_slider(fragment_type, fragment_range, select_random, cont):
    fragment_def = random.choice(fragment_range) if select_random else fragment_range[0]
    return cont.select_slider(f"Select an {fragment_type}", fragment_range, value=fragment_def)


def fragment_selection():
    st.markdown("#### 0.1 Metadata Selection")
    if not DEBUGGING:  # choose metadata subset from which the interview can be selected by user
        metadata = global_analysis.filter_metadata()
        metadata.to_csv(METADATA_FILE_PATH)
    metadata = pd.read_csv(METADATA_FILE_PATH, index_col="Participant_ID")

    st.markdown("#### 0.2 Fragment Selection")
    st.write(f"Ensure that transcripts and metadata have been processed with sample hop: {PARAMS['sample_hop']}")
    dwd = daic_woz_datasets.DaicWozDataset(METADATA_FILE_PATH)  # load Daic Woz dataset
    select_random = st.button("Select random fragment")  # if True the fragment will be selected randomly
    segment_id, sample_id = None, None

    # Interview selection
    interview_id = fragment_slider("interview", sorted(metadata.index), select_random, st)
    interview = custom_classes.Interview(interview_id)
    transcript = interview.get_transcript()
    fragment = interview  # set participant class as the fragment to analyse

    # Segment selection
    col1, col2 = st.columns((0.2, 0.8))
    if col1.checkbox("Use segment", value=True):
        segs_interview = list(range(transcript.shape[0]))
        segment_id = fragment_slider("segment", segs_interview, select_random, col2) if len(segs_interview) > 1 else 0
        segment = custom_classes.Segment(interview_id, segment_id)
        fragment = segment  # set selected segment class as the fragment to analyse

        # Sample selection
        c1, c2 = st.columns((0.2, 0.8))
        if c1.checkbox("Use sample", value=True):
            n_samples_segment = int(segment.segment_data["last_sample"] - segment.segment_data["first_sample"])
            samples_segment = list(range(0, n_samples_segment))
            sample_id = fragment_slider("sample", samples_segment, select_random, c2) if len(samples_segment) > 1 else 0
            sample = custom_classes.Sample(interview_id, segment_id, sample_id)
            fragment = sample  # set selected sample as the fragment to analyse

    # display basic fragment data
    idx = dwd.get_idx(interview_id, segment_id if segment_id else 0, sample_id if sample_id else 0)
    st.write(f"Fragment data:")
    st.write(f"- DaicWozDataset idx: **{idx}**")
    st.write(f"- Participant gender: **{'male' if interview.participant.gender else 'female'}**")
    dep_str = "(depressed)" if fragment.participant.label else "(not depressed)"
    st.write(f"- Participant PHQ8 score: **{fragment.participant.score} {dep_str}**")
    return fragment, transcript


def display_model_deployment(fragment):
    st.write("### Model Deployment")
    # load user selected model
    model_path, model_checkpoint = global_analysis.select_model(st.columns(2), MODELS_PATH)
    model = train_model.load_model(DEVICE, model_path, model_checkpoint)
    # load fragment data
    spec_fragment = fragment.get_spectrogram(transpose=True).unsqueeze(0)
    target = fragment.interview.participant.label
    try:
        # get and display model results
        pred_probability = model(spec_fragment)  # get model prediction
        st.write(f"##### - Output depression prob: *{round(pred_probability.item(), 4)}*")
        st.write(f"##### - Target label: *{round(target, 4)}*")
        loss = nn.BCELoss()(pred_probability, torch.tensor(target).float().to("cuda")).item()
        st.write(f"##### - Loss: *{round(loss, 4)}*")
    except:
        st.warning("There will be an error if you don't select a model with the current cnn architecture")


def fragment_audio_analysis(fragment, transcript):
    st.write("#### 1.1 Audio subset selection")
    st.write("Select a fragment subset. ")
    # give the choice to use transcript times if the fragment being analysed is an interview
    fragment_is_interview = type(fragment) == custom_classes.Interview
    c2, use_transcript_times = st.container(), False
    if fragment_is_interview:
        c1, c2 = st.columns([0.3, 0.7])
        use_transcript_times = c1.checkbox("Use transcript times in range slider")

    if use_transcript_times:  # time range of audio selection widget depending on transcript times
        interview_duration = fragment.audio_times[1] - fragment.audio_times[0]  # get interview duration in seconds
        times = [0, ] + list(transcript["start_time"]) + [interview_duration, ]
        t = c2.select_slider("Select fragment subset", times, format_func=lambda x: secs2time(x, return_str=True),
                             value=[times[len(times) * 4 // 10], times[len(times) * 6 // 10]])
    else:  # free time range of audio selection widget
        audio_start, audio_end = secs2time(fragment.audio_times[0]), secs2time(fragment.audio_times[1])
        t = c2.slider("Select fragment subset", step=timedelta(seconds=1), min_value=audio_start,
                      max_value=audio_end, value=(audio_start, audio_end), format="mm:ss")
        t = [t_i.minute * 60 + t_i.second for t_i in t] if t != (audio_start, audio_end) else fragment.audio_times
    t_format = secs2time(t[0]).strftime('%M:%S'), secs2time(t[1]).strftime('%M:%S')

    st.markdown("### Interview Audio Analysis")
    # load and display the chosen time range from the selected audio file
    st.markdown(f"Interview audio from participant {fragment.participant_id} from {t_format[0]} to {t_format[1]}:")
    fragment_audio, sr = fragment.get_audio(times=t)
    fragment_audio_arr = fragment_audio.cpu().numpy()[0]
    display_interactive_audio(fragment_audio_arr, sr)

    # visualize waveform if desired
    if st.checkbox("Show audio waveform", value=True):
        st.pyplot(fragment.show_waveform(times=t)[0])

    # visualize spectrogram if desired
    if st.checkbox("Show spectrogram", value=True):
        cont, plot_container, cols = st.container(), st.container(), st.columns((.2, .2, .2, .2, .2))
        # stft transform selection
        stft_mel = cols[0].checkbox("Mel Spectrogram", value=True)
        # normalization type selection
        norm_types = ["globally", "locally", "Not normalize"]
        default_norm = int(PARAMS["norm_spec_locally"])
        norm = cols[1].radio("Normalization type", norm_types, index=default_norm)
        norm = None if norm == "Not normalize" else norm
        # spec display options selection
        transpose, color_bar = cols[2].checkbox("Transpose"), cols[3].checkbox("Add color bar")
        # display spec with the selected parameters
        stft_show_params = {"use_mel": stft_mel, "normalize": norm, "transpose": transpose, "add_color_bar": color_bar}
        spec_plot, spec = fragment.show_spectrogram(audio_times=t, **stft_show_params)
        spec_type, spec_mean, spec_var = 'Mel' if stft_mel else '', spec.mean().round(), spec.var().round()
        cont.write(f"The {spec_type} Spectrogram have dims: {spec.shape}, mean: {spec_mean} and var: {spec_var}")
        plot_container.pyplot(spec_plot)


def fragment_model_deployment(fragment):
    pass  # TODO


def app():
    st.markdown("## Individual Interview Analysis")

    # let fragment be selected by user
    st.markdown("### 0. Data Selection")
    fragment, transcript = fragment_selection()

    st.write("### 1. Fragment audio analysis")
    if st.checkbox("Display fragment audio analysis"):
        fragment_audio_analysis(fragment, transcript)

    st.write("### 2. Interview transcript analysis")
    # display transcript analysis if the fragment is an interview
    if type(fragment) == custom_classes.Interview:
        st.markdown("### Interview Transcript Analysis")
        metadata_cont = st.container()
        # filter transcript df by selected time range
        transcript = transcript.loc[transcript["stop_time"] >= t[0]]
        transcript = transcript.loc[transcript["start_time"] <= t[1]]
        # display turn times distribution
        transcript["turn_time"] = transcript["stop_time"] - transcript["start_time"]
        fig = px.histogram(transcript, x="turn_time", y="turn_time", nbins=30, histfunc='sum', title="Speaking Time")
        st.plotly_chart(fig, use_container_width=True)

        # replace time cols in transcript df to gain interpretability
        transcript["start_time"] = transcript["start_time"].apply(lambda x: secs2time(x, return_str=True))
        transcript["stop_time"] = transcript["stop_time"].apply(lambda x: secs2time(x, return_str=True))

        metadata_cont.write(transcript)  # display resultant df

    # display segment text if fragment is not an interview
    else:
        if type(fragment) == custom_classes.Segment:
            segment_text = fragment.interview.get_transcript(fragment.audio_times)["value"].item()
            st.write("**Fragment text**:")
        else:
            segment_text = fragment.interview.get_transcript(fragment.segment.audio_times)["value"].item()
            st.write("**Fragment text** (sample text must be contained on it):")
        st.write(segment_text)

    """
    st.write("2. Fragment model deployment")
    if st.checkbox("Display fragment model deployment"):
        fragment_model_deployment(fragment)
    """


if __name__ == "__main__":
    # change to main directory
    project_dir = Path(__file__).resolve().parents[3]
    os.chdir(project_dir)

    app()
