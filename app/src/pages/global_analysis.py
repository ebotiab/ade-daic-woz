import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
from collections import Counter
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataprep.eda import create_report
import streamlit.components.v1 as components

from src.hyperparameters import PARAMS
from src.features import balance_train_data
from src.models import train_model, predict_model, daic_woz_datasets
from src.models.daic_woz_datasets import DaicWozDatasetReport
from app.src import constants, utils


# executing script mode (change if you want to debug the code), since st session state does not work when debugging
DEBUGGING = False
if DEBUGGING:
    project_dir = Path(__file__).resolve().parents[3]
    os.chdir(project_dir)

# folder paths (change if the project folder structure is different)
REPORTS_DIR = "reports"
METADATA_PATH = "data/interim/labels"
TRANSCRIPTS_FOLDER = "data/interim/transcripts"
MODELS_PATH = "models"
# device where model will be deployed (change if necessary)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def filter_col_value(df, col_name, container, label):
    """
    display radio widgets and filter df according selected option
    """
    options = ["All", ] + list(df[col_name].unique())
    col_value = container.radio(label=label, options=options, key=col_name)
    if col_value == "All":
        return df
    if is_numeric_dtype(df[col_name]):
        col_value = 1 if col_value == "Yes" else 0
    return df[df[col_name] == col_value]


def filter_phq8_score(df, container):
    """
    display slider widget and filter df according selected range
    """
    min_score = 10 if st.session_state["PHQ8_Binary"] == "Yes" else 0
    max_score = 10 if st.session_state["PHQ8_Binary"] == "No" else 24
    score_range = container.slider("Filter by depression severity", min_score, max_score, value=[min_score, max_score])
    return df[df["PHQ8_Score"].between(*score_range)]


def plot_distributions(df, col_name1, col_name2, container):  # TODO: convert to inputs
    """
    generate and display bar plot with class percentages
    """
    df_g = df.groupby([col_name1, col_name2]).size().reset_index()
    df_g['percentage'] = df.groupby([col_name1, col_name2]).size().groupby(level=0).apply(
        lambda x: 100 * x / float(x.sum())).values
    df_g.columns = [col_name1, col_name2, 'Counts', 'Percentage']
    if st.session_state["perc"]:
        df_g["Counts"] /= df.shape[0]
    fig = px.bar(df_g, x=col_name1, y=['Counts'], color=col_name2, text=df_g['Percentage'].apply(
        lambda x: '{0:1.2f}%'.format(x)))
    if st.session_state["perc"]:
        fig.update_layout(yaxis_title="Percentage")
    container.plotly_chart(fig, use_container_width=True)


def ensure_one_partition():
    if not DEBUGGING:
        if st.session_state["train"] + st.session_state["dev"] + st.session_state["test"] == 0:
            st.session_state["train"] = True


def filter_metadata():
    """
    filter data according selected values in widgets
    """
    st.write(constants.filter_data_info)
    cols = st.columns([0.1, 0.1, 0.15, 0.5])
    # filter by partitions
    partitions_dict = {}
    for partition in ["train", "dev", "test"]:
        if cols[0].checkbox(partition, value=partition == "train", key=partition, on_change=ensure_one_partition):
            partition_path = Path(METADATA_PATH, f"{partition}_metadata.csv")
            partitions_dict[partition] = pd.read_csv(partition_path, index_col="Participant_ID")
    metadata = pd.concat(partitions_dict.values(), axis=0)
    # rename values in metadata binary variables
    metadata["PHQ8_Binary"] = metadata["PHQ8_Binary"].replace([0, 1], ["not depressed", "depressed"])
    metadata["Gender"] = metadata["Gender"].replace([0, 1], ["female", "male"])
    # filter data by metadata variables
    initial_num_rows = metadata.shape[0]
    for c, l, ct in zip(["Gender", "PHQ8_Binary"], ["Filter by gender", "Filter by depressed"], cols[1:3]):
        metadata = filter_col_value(metadata, c, ct, l)
    metadata = filter_phq8_score(metadata, cols[3])
    # display data about the filtering
    st.write(f"Original number of instances in {', '.join(partitions_dict.keys())} "
             f"partition{'s' if len(partitions_dict)>1 else ''}: {initial_num_rows}")
    st.write(f"Number of instances in generated data: **{metadata.shape[0]}** "
             f"({initial_num_rows - metadata.shape[0]} have been filtered out)")
    # convert labels back to integers to model deployment section
    metadata["PHQ8_Binary"] = metadata["PHQ8_Binary"].replace(["not depressed", "depressed"], [0, 1]).astype(int)
    return metadata


def show_bal_reduction_changes(old_metadata, new_metadata, message, class_type_reduction=False):
    st.write(message)
    if class_type_reduction:
        to_display = balance_train_data.reduction_changes_str(old_metadata, new_metadata, class_type_reduction)
        st.write(". ".join(to_display[:3])), st.write(to_display[3]), st.write(". ".join(to_display[4:]))
    else:
        st.write(". ".join(balance_train_data.reduction_changes_str(old_metadata, new_metadata, False)))


def balance_metadata(metadata):
    # select balance options
    cols = st.columns(3)
    hop_bal_level = cols[0].number_input("Hop Balance level", 0., 1., PARAMS["hop_balance_level"])
    hop_bal_index = cols[0].number_input("Hop Balance index", 0, len(metadata), PARAMS["hop_balance_index"])
    ids_und_level = cols[1].number_input("Ids Undersampling", 0., 1., PARAMS["ids_undersampling_level"])
    class_und_level = cols[1].number_input("Class Undersampling", 0., 1., PARAMS["class_undersampling_level"])
    ids_filter_level = cols[2].number_input("Ids few samples filter", value=PARAMS["ids_few_samples_filter_level"])

    st.write("#### Reduction changes in metadata")
    # balance ids with undersampling
    old_metadata = metadata.copy()
    metadata = balance_train_data.hop_balance(metadata, hop_bal_level, hop_bal_index, TRANSCRIPTS_FOLDER)
    show_bal_reduction_changes(old_metadata, metadata, "- **Hop Balance**:")

    # balance ids with undersampling
    old_metadata, metadata = metadata.copy(), balance_train_data.ids_undersampling(metadata, ids_und_level)
    show_bal_reduction_changes(old_metadata, metadata, "- **Ids undersampling**:")

    # balance classes with undersampling
    old_metadata, metadata = metadata.copy(), balance_train_data.class_undersampling(metadata, class_und_level)
    # st.write(f"{metadata.shape}, {class_und_level}, ")
    show_bal_reduction_changes(old_metadata, metadata, "- **Class undersampling**:", True)

    # filter ids with few samples
    old_metadata, metadata = metadata.copy(), balance_train_data.ids_few_samples_filter(metadata, ids_filter_level)
    show_bal_reduction_changes(old_metadata, metadata, "- **Ids few samples filter**:")
    return metadata


@st.cache
def get_dataloader_data(balance_type, balanced_metadata, data_loader_params, n_batches):
    # load dataset
    dataset_params = {"metadata": f"data/processed/labels/global_analysis.csv"}
    dwd = DaicWozDatasetReport(**dataset_params)
    dwd.metadata = balanced_metadata
    # create weighted sampler if selected
    weighted_sampler = None
    if balance_type != "no_balance":
        weighted_sampler = train_model.create_weighted_sampler(dwd, balance_type)
    data_loader_params["sampler"] = weighted_sampler
    # load dataloader
    train_dataloader = DataLoader(dwd, **data_loader_params)
    # get dataloader data
    n_batches = min(n_batches, len(train_dataloader) - 1)
    labels_all, ids_total = [], []
    for bath_counter, (labels, p_ids, idx) in enumerate(train_dataloader):
        labels_all += labels.tolist()
        ids_total += p_ids.tolist()
        if bath_counter == n_batches:
            break
    return labels_all, ids_total, n_batches, dwd


def display_ids_distribution(ids_array, st_container, title):
    ids_df = pd.DataFrame({"id": list(ids_array), "value": ids_array.values()})
    st_container.plotly_chart(px.bar(ids_df, x="id", y="value", title=title), use_container_width=True)
    st_container.write("**Statistics:**")
    st_container.write(f"- Number of participants: {ids_df.shape[0]}")
    st_container.write(f"- Value Mean: {round(ids_df['value'].mean(), 2)}")
    st_container.write(f"- Value Std: {round(ids_df['value'].std(), 2)}")


def dataloader_analysis(metadata):
    # get dataloader data
    cols = st.columns(2)
    n_batches = cols[0].number_input("Batch where you want to stop", min_value=1, value=PARAMS["max_batches"])
    batch_size = cols[0].number_input("Batch size", min_value=1, value=int(PARAMS["batch_size"]))
    bal_types = ["no_balance", "ids", "classes", "classes_ids"]
    balance_type = cols[1].select_slider("Balance type", bal_types, value=(PARAMS["weights_balance_type"]))
    d_params = {"batch_size": batch_size, "shuffle": balance_type == "no_balance"}
    labels_all, ids_total, n_batches, dwd = get_dataloader_data(balance_type, metadata, d_params, n_batches)

    # display class dataloader data
    st.write("#### Dataloader obtained data")
    st.write(f"**Number of batches: {n_batches}**")
    st.write("To be balanced, label mean should approximate 50%:")
    st.write("Label mean:", np.mean(labels_all).round(2) * 100, "%")
    # display ids dataloader data
    st.write("To be balanced, ids by class distribution should approximate an Uniform Distribution and var. be small:")
    if st.checkbox("Separate analysis by class"):
        for i_class, col in zip([1, 0], st.columns(2)):
            ids_class = Counter([i for i, l in zip(ids_total, labels_all) if l == i_class])
            title = "Depressed samples ids" if i_class else "Not Depressed samples ids "
            display_ids_distribution(ids_class, col, title)
    else:
        display_ids_distribution(Counter(ids_total), st, "All samples ids")


def plot_dist_histograms(x1, x2, agg_mode, is_cumulative, seg_by_class, title, n_bins):
    fig = go.Figure()
    if seg_by_class:
        fig.add_trace(go.Histogram(x=x1, y=x1, histfunc=agg_mode, histnorm='probability',
                                   name="depressed", cumulative_enabled=is_cumulative, nbinsx=n_bins))
        fig.add_trace(go.Histogram(x=x2, y=x2, histfunc=agg_mode, histnorm='probability',
                                   name="not depressed", cumulative_enabled=is_cumulative, nbinsx=n_bins))
    else:
        fig.add_trace(go.Histogram(x=x1+x2, y=x1+x2, histfunc=agg_mode,
                                   histnorm='probability', cumulative_enabled=is_cumulative, nbinsx=n_bins))
    fig.update_layout(barmode='group')
    cumulative_label = "cumulative" if is_cumulative else "non cumulative"
    fig.update_layout(title_text=f"{title} {cumulative_label} distribution using {agg_mode} aggregation")
    st.plotly_chart(fig, use_container_width=True)
# TODO: Boxplot function


def metadata_analysis(metadata):
    metadata = metadata.drop(["last_sample", "first_sample"], axis=1)
    # display selected metadata subset
    st.markdown(f"Selected metadata has dims {metadata.shape}:")
    st.write(metadata)

    st.markdown("#### 2.1. Number of samples")
    # display number of samples distribution
    metadata["index"] = metadata.index
    fig = px.histogram(metadata, x="index", y="num_samples", nbins=200, color="PHQ8_Binary",
                       labels={"index": "interview", "num_samples": "number of samples"})
    st.plotly_chart(fig, use_container_width=True)
    # display number of samples distribution statistics
    st.write("Distribution statistics:")
    utils.statistics_pd_col(metadata["num_samples"])

    # generate and display plotly figures with the class
    st.markdown("### 2.2. Class Distributions")
    st.checkbox("Show Y axis as percentage", key="perc")
    plot_col_names = ["PHQ8_Binary", "Gender"]
    # probability distributions bar plots
    containers = st.columns((0.3, 0.3, 0.6))
    for col1, col2, cont in zip(plot_col_names, reversed(plot_col_names), containers[:2]):
        plot_distributions(metadata, col1, col2, cont)
    # disorder severity boxplot comparing gender
    fig = px.box(metadata, x="Gender", y="PHQ8_Score", points="all")
    containers[2].plotly_chart(fig, use_container_width=True)

    # generate and display dataprep report
    report = create_report(metadata, title="metadata report")
    report_path = str(Path(REPORTS_DIR, "reports.html"))
    report.save(path=report_path)
    html_file = open(report_path, 'r', encoding='utf-8')
    source_code = html_file.read()
    st.markdown("### 2.3. Dataprep report")
    st.markdown("In this report the correlations between pair of vars and individual distributions can be observed")
    components.html(source_code, height=500, scrolling=True)


def transcript_analysis(metadata):
    st.markdown("### Transcript Analysis")

    # segment num samples analysis
    st.markdown("#### 3.1. Number of samples")
    # display widgets to select type of histogram
    col1, col2, col3, col4, col5 = st.columns(5)
    is_cumulative = col1.checkbox("Cumulative histogram", value=True)
    by_class = col1.checkbox("Separate by class", value=True)
    agg_mode = col2.radio("Aggregation type", ["sum", "count"])
    hist_components_label = col3.radio("Histogram components", ["segments", "mean per interview", "sum per interview"])
    hc = ["segments", "mean per interview", "sum per interview"].index(hist_components_label)
    thresh1 = col4.number_input("Threshold 1st histogram", value=100)
    n_bins1 = col4.number_input("Nº of bins 1st histogram (if 0 use default)", value=0)
    thresh2 = col5.number_input("Threshold 2nd histogram", value=1000)
    n_bins2 = col5.number_input("Nº of bins 2nd histogram (if 0 use default)", value=0)
    # save speaking times in arrays according class
    turn_time_dep, turn_time_norm = [], []
    turn_samples_dep, turn_samples_norm = [], []
    for participant_id in metadata.index:
        f = f"{participant_id}_TRANSCRIPT.csv"
        transcript = pd.read_csv(Path(TRANSCRIPTS_FOLDER, f))
        #  save speaking time in lists depending on label
        t_turns = list(transcript["stop_time"] - transcript["start_time"])
        # t_turns = list(t_turns[t_turns < thresh1])
        t_samples = list(transcript["num_samples"])  # [transcript["num_samples"] < thresh2])
        if metadata.loc[int(f[:3])]["PHQ8_Binary"] == "depressed":
            turn_time_dep += t_turns if hc == 0 else [np.mean(t_turns), ] if hc == 1 else [sum(t_turns), ]
            turn_samples_dep += t_samples if hc == 0 else [np.mean(t_samples), ] if hc == 1 else [sum(t_samples), ]
        else:
            turn_time_norm += t_turns if hc == 0 else [np.mean(t_turns), ] if hc == 1 else [sum(t_turns), ]
            turn_samples_norm += t_samples if hc == 0 else [np.mean(t_samples), ] if hc == 1 else [sum(t_samples), ]
    turn_time_dep, turn_time_norm = [[j for j in i if j < thresh1] for i in [turn_time_dep, turn_time_norm]]
    turn_samples_dep, turn_samples_norm = [[j for j in i if j < thresh2] for i in [turn_samples_dep, turn_samples_norm]]
    # display the speaking time distribution figure
    plot_dist_histograms(turn_time_dep, turn_time_norm, agg_mode, is_cumulative, by_class, "Speaking time", n_bins1)
    plot_dist_histograms(turn_samples_dep, turn_samples_norm, agg_mode, is_cumulative, by_class, "Num samples", n_bins2)


def audio_analysis(metadata):
    audios_speaking_duration, segments_duration = [], []
    for participant_id in metadata.index:
        f = f"{participant_id}_TRANSCRIPT.csv"
        transcript = pd.read_csv(Path(TRANSCRIPTS_FOLDER, f))
        audios_speaking_duration.append((transcript["stop_time"] - transcript["start_time"]).sum())
        segments_duration += list((transcript["stop_time"] - transcript["start_time"]))
    audios_speaking_duration, segments_duration = np.array(audios_speaking_duration), np.array(segments_duration)
    statistics_df = pd.DataFrame({"fragment type": ["speaking total duration", "speaking turn duration"]})
    operations = {"N": len, "mean": np.mean, "std": np.std, "min": np.min, "max": np.max}
    for k in operations:
        statistics_df[k] = operations[k](audios_speaking_duration), operations[k](segments_duration)
    st.write(statistics_df)


def select_model(st_c, models_p):
    model_path = st_c[0].selectbox("Select model", sorted(Path(models_p).iterdir()), format_func=lambda x: str(x)[7:])
    last_checkpoint = int(str(sorted(Path(model_path).iterdir()))[-9:-7])
    model_checkpoint = st_c[1].number_input("Select model checkpoint", max_value=last_checkpoint, value=last_checkpoint)
    return model_path, model_checkpoint


def model_deployment(metadata):
    if not DEBUGGING:  # give a warning if test has not been selecting when filtering dataset
        _ = st.warning("You are not evaluating the test set") if not st.session_state["test"] else None
    # let the user select the model, checkpoint, loss type and if it is desired to separate predictions by sex
    cols = st.columns(3)
    model_path, model_checkpoint = select_model(cols, MODELS_PATH)
    by_sex = ["female", "male"] if cols[2].checkbox("Separate by sex") else ["all"]
    if st.checkbox("Execute testing process"):
        # get model predictions (in pred_metadata dataframe) and the obtained metrics when testing the model
        pred_metadata = compute_pred_cached(model_path, model_checkpoint, metadata)
        # display results to the user
        for i in range(len(by_sex)):  # display sample prediction distribution by sex (if desired) and by class
            df_seg_1 = pred_metadata.loc[pred_metadata["Gender"] != by_sex[-(i+1)]]
            metrics = predict_model.compute_test_metrics(df_seg_1, DEVICE)
            for j, c in zip([0, 1], st.columns(2)):
                label = "depressed" if j else "not depressed"
                df_seg_2 = df_seg_1.loc[df_seg_1["PHQ8_Binary"] == j]
                utils.display_distribution(df_seg_2, "index", ["n_pred_dep", "n_pred_norm"], c, by_sex[i]+" "+label)
                _ = c.write(f"{'Sample' if j==0 else 'Participant'} metrics"), c.write(metrics[j])  # display metrics
        st.write(pred_metadata)


@st.cache
def compute_pred_cached(model_path, model_checkpoint, metadata):
    # load model from user selection
    model = train_model.load_model(DEVICE, model_path, model_checkpoint)
    # build dataset from current metadata
    dwd = daic_woz_datasets.DaicWozDataset(metadata)
    # get testing obtained global metrics
    predict_model.compute_predictions(dwd, model, DEVICE)
    return dwd.metadata


def app():
    st.write(constants.global_intro)

    st.markdown("## 0. Data Selection")
    st.markdown("### 0.1 Filter dataset")
    if not DEBUGGING:
        metadata = filter_metadata()  # let user select metadata subset
        metadata.to_csv(Path(METADATA_PATH, "global_analysis.csv"))
    metadata = pd.read_csv(Path(METADATA_PATH, "global_analysis.csv"), index_col="Participant_ID")
    st.markdown("### 0.2 Balance dataset")
    if st.checkbox("Display data balance options"):
        metadata = balance_metadata(metadata)

    st.markdown("## 1. Metadata Analysis")
    if st.checkbox("Display metadata analysis"):
        metadata_analysis(metadata)

    st.markdown("## 2. Model deployment")
    if st.checkbox("Display model deployment"):
        model_deployment(metadata)


"""
    st.write("## 2. Transcripts analysis")
    if st.checkbox("Display transcripts analysis"):
        transcript_analysis(metadata)

    st.markdown("## 3. Audio analysis")
    if st.checkbox("Display audio analysis"):
        audio_analysis(metadata)
"""


if __name__ == "__main__":
    app()
