from src.hyperparameters import PARAMS

filter_col_names = ["Gender", "AVECParticipant_ID", "PHQ_Binary", "PCL-C (PTSD)"]
filter_labels = ["Select gender", "Select partition", "Filter by depressed", "Filter post-traumatic"]
slider_labels = ["Filter by depression severity", "Filter by PTSD severity"]


# STREAMLIT APP TEXT

title = "DAIC-WOZ Automatic Depression Detection"

# 1. Project Introduction

"""
This project is part of a larger effort which is the FDJ-TMC research. A study that aims to provide the physicians with 
technological tools to assist them with the follow-up of patients diagnosed with depression or anxiety, based on audio 
samples recorded with certain periodicity.
"""

intro_content = """
## Project Introduction
The aim of this project is to implement an end-to-end system for automatic depression estimation based on speech 
using DAIC-WOZ database.

This application allows some exploratory data analysis in an user-friendly interactive way, as well as sections where 
the model results are shown for a user selected interview.

You can explore the different pages though the select box from the sidebar
"""


# 2. Global Analysis

global_intro = """
# Global Analysis
In this page an analysis of the different data types (metadata, audios and transcripts) is presented.
"""

filter_data_info = """
Here you can select the partition and filter the data you want to analyse. If multiple partitions are 
selected, the corresponding dataframes will be concatenated.
"""

# 3. Preprocess Report

preprocess_report_intro = """
## Preprocess report
The preprocess is the stage where the raw data is transformed so it can be processed by the designed predictive model. 
It also is used to create a new dataset from which the model can learn more efficiently and generate more accurate 
predictions. This page explains the different steps in the preprocess performed for this project.
"""

segment_extraction_1 = """
The model is designed to receive as input a spectrogram where the participant (not the interviewer) is speaking. We can 
use the transcripts data to extract this information, since they provide the start and stop times in the audio for each 
of the sentences as well as the speaker identifier. To achieve this, it is necessary to apply a series of
transformations to the original transcripts.

This pipeline is described in the following, taking as an example the interview 345 and its corresponding transcript:
"""

n_sample_extraction_1 = f"""
### 2. Number of samples extraction
It is required that the spectrogram samples are {PARAMS["sample_len"]} frames long, which is equivalent to 
{PARAMS["sample_time"]} secs of  audio. Rather than first load the samples and then pass them all to the model, 
the approach followed consists on the creation of a Dataset Pytorch class, that pass the samples to the model just after
of being created. This has the advantage of be able to perform all computations in parallel using a cuda device.

To create a Dataset class we need to design mapping from a given index to a spectrogram sample. One way to achieve 
this would be to iterate from all samples for each segment until the index is reached. To make this process 
efficient it is needed to extract the number of samples that can be created for each segment and then for each 
interview.
"""

n_sample_extraction_2 = f"""
The waveform and the spectrogram can be segmented into samples of {PARAMS["sample_time"]} seconds and a selected sample
can be highlighted.
"""

# 4. Training Report

training_report_intro = """
## Training report
Once we the data has been preprocessed, we can pass the new dataset to the deep learning model, so it starts the 
training process. 
"""

hop_balance_report = r"""
### 1. Hop Balance
Hop id balance is a designed technique that balance sample ids (to the model is not to be biased to participant 
specific characteristics). It consists in setting the sample hop of each interview so that all of them have the same 
number of samples. 

The number of samples in one segment or participant speaking turn can be defined by the following equation:
$$
N = \lceil \frac{(b-p-4)}{h} \rceil
$$
Where $b_s$ and $p_s$ are times where the segment starts and stops, respectively.

Therefore, the number of samples in one interview can be defined by:
$$
I = \sum_{i \in S}\lceil \frac{(b_i-p_i-4)}{h} \rceil
$$
Where $S$ is the set of segments for that interview.

So, if we set the number of samples of one interview, say $I_m$, we can obtain its corresponding sample hop by 
approximately:
$$
h_m =  \sum_{i \in S_x} \frac{(b_i-p_i-4)}{I_m}
$$

In the designed technique, to completely balance ids and maximize the generated data, $I_m$ corresponds with the 
interview with the lowest number of samples when the sample hop is equal to one. The hop balance steps are:
1. Get the processed transcripts and interviews with the sample hop equal to 1.
2. Find the interview with the lowest number of samples
3. Compute the sample hop per each interview according to the equations and include it as a new metadata variable
The DaicWozDataset implementation must be adapted to use the specified sample hop per each interview.
"""  # TODO: deal with approximations in the math expressions

dataset_1 = """
### Dataset
Load the preprocessed dataset that it is going to be passed to the model
"""

data_loader_1 = r"""
### Sampling from dataloader
Since we need to pass to the model balanced classes, the Dataloader torch class is created by passing weights to
each sample.

The weights are computed according with the following formula:

$$
W_x = \frac{NT}{N_cN_{x_p}}, x\in(0,NT), c\in(0,1)
$$

Where $W_x$ is the weight corresponding to the sample $x$, $NT$ is the total number of samples, $N_c$ is the number
of samples that belong to the class $c$ and $N_{x_p}$ is the number of samples that correspond to the participant $x_p$,
which corresponds with the interview where the sample $x$ belong.

**Dataloader data**:
"""

data_loader_2 = """
### Sampling from dataloader
The TwoParticipant dataset consists on a dataset with only two participants with different classes and the same
number of samples per each of them.

**Dataloader data**:
"""
