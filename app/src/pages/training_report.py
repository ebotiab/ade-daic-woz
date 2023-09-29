import os
from pathlib import Path
import streamlit as st


from app.src import constants
from src.hyperparameters import PARAMS
from src.models.daic_woz_datasets import DaicWozDatasetReport


# TODO: create hash for dataloader and add colors by class in distributions
def app():
    # INTRO
    st.write(constants.training_report_intro)

    # HOP BALANCE
    # st.write(constants.hop_balance_report)

    # DATASET SECTION
    st.write(constants.dataset_1)
    cols = st.columns(2)
    partition = cols[0].selectbox("Select partition", ["test", "train", "dev", "train_dev"])
    dataset_params = {"metadata": f"data/processed/labels/{partition}_metadata.csv"}
    dwd = DaicWozDatasetReport(**dataset_params)
    metadata = dwd.metadata
    st.write(metadata.shape)
    st.write(f"With a sample hop of **{PARAMS['sample_hop']}**,"
             f" the **{partition}** partition has **{len(dwd)}** samples available")

    # DATALOADER SECTION




if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[3]
    os.chdir(project_dir)

    app()
