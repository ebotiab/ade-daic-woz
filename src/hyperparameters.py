import librosa

"""
In this script the hyper-parameters can be changed to conduct different experiments.

IMPORTANT TO REMEMBER:
1. Run build_processed_transcripts.py, build_processed_metadata.py and balance_train_data.py scripts when changing any 
of the hyperparameters in PREPROCESS_PARAMS (in that order)
2. Run get_specs_feature.py script when changing any parameter that can affect the spectrograms created 
3. Change 'experiment_name' in each experiment (which must be set without spaces since it is also used as a file name)
"""

# hyperparameters that require computations from previous ones
TIME_SEP_SPEAKER = 2
SAMPLE_TIME = 4

SR = 16000
STFT = {"n_fft": 1024, "hop_length": 512}
# STFT = {"n_fft": 400, "hop_length": 200} # default torchaudio stft params
# STFT = {"n_fft": 4096, "hop_length": 2048}  # paper stft params

FIRST_FRAME_IND = librosa.time_to_frames(0, sr=SR, **STFT)
SAMPLE_N_AUDIO_FRAMES = librosa.time_to_samples(SAMPLE_TIME, sr=SR)  # convert time to audio frames
SAMPLE_LEN = librosa.time_to_frames(SAMPLE_TIME, sr=SR, **STFT) - FIRST_FRAME_IND  # convert time to spec frames

SAMPLE_HOP = 20  # number of spectrogram frames that separate any pair of contiguous spectrogram samples
SAMPLE_HOP_AUDIO_FRAMES = SAMPLE_HOP*STFT["hop_length"]  # convert spec frames to audio frames
SAMPLE_HOP_TIME = librosa.samples_to_time(SAMPLE_HOP_AUDIO_FRAMES, sr=SR)  # convert audio frames to time

PREPROCESS_PARAMS = {
    # Required to rerun build_processed_transcripts and build_processed_metadata if changed:
    "use_silences": False,
    "time_sep_speaker": TIME_SEP_SPEAKER,
    "sample_time": SAMPLE_TIME,
    "sample_n_audio_frames": SAMPLE_N_AUDIO_FRAMES,
    "sample_len": SAMPLE_LEN,
    "sr": SR,
    "stft": STFT,
    "first_frame_ind": FIRST_FRAME_IND,
    "sample_hop": SAMPLE_HOP,
    "sample_hop_audio_frames": SAMPLE_HOP_AUDIO_FRAMES,
    "sample_hop_time": SAMPLE_HOP_TIME,
}

# TODO: IMPLEMENT SIGMOID FOCAL LOSS
# TODO: "batches_min_id_samples": True,  # compute max nº batches as smallest nº of samples in the selected interviews
PARAMS = {
    # Not required to rerun any script if changed:
    "pretrained_model_name": "",  # train new model from weights stored in this model, if exists
    "model_name": "lr_decay",
    "experiment_description": "",
    "random_seed": 7,  # used to obtain random sample, if it is set to 0 not random (random recommended if sample_hop>1)
    "concat_train_dev": False,  # use for train the model a concatenation of train and dev sets and validate with test
    "lr_rate": 1e-3,
    "lr_decay_freq": 2,  # number of epochs to apply lr decay, set to -1 to not apply it
    "lr_decay_factor": 0.9,  # factor that is multiplied by the lr each lr_decay_freq epochs
    "dep_train_threshold": 10,  # new depression threshold to be set on training set to balance classes (lower than 10)
    "batch_size": 20,
    "n_epochs": 10,
    "max_batches": 50,
    "norm_spec_locally": False,  # normalize the spectrograms locally or globally
    "ids_undersampling_level": 0.,  # level of undersampling to balance classes (from 0 to 1)
    "class_undersampling_level": 1.,  # level of undersampling to balance classes (from 0 to 1)
    "class_bal_max_data": True,
    "ids_few_samples_filter_level": 0.,  # level for removing interviews with  (from 0 to 1)
    "weights_balance_type": None,  # balance technique to apply in the dataset, it can take values from:
    # - None: don't apply any balance technique  with weights in the dataloader
    # - classes: balance only the classes using weights
    # - classes_ids: predefined weights will be used to balance classes and ids for each of the classes,
    # - ids: balance only the classes using weights (recommended to balance then classes by undersampling)
    "use_focal_loss": False,  # use sigmoid focal loss instead of bce loss
    "transpose_spec": True,  # transpose spec to compute convolutions in frequency axis
    "evaluate_training": False,
    "filter_by_sex": None,
    # hop balance: use different sample hop per interview in order to balance ids (use then und/ovs to balance classes)
    "hop_balance_level": 0.,  # hop balance level
    "hop_balance_index": 0,  # interview index (ordered by num_samples) from which perform hop_balance

    # plot/display configurations
    "plot_metrics_frequency": 2,  # number of epochs to display training and evaluation metrics

    # Required to rerun build_processed_transcripts and build_processed_metadata if changed:
    **PREPROCESS_PARAMS,

    # Required to run get_features again:
    "stft_mel": True,
    "n_mels": 40,  # (only used if stft_mel is set to True)
    "get_features_max_batches": 1000,  # (the bigger, the more precise will be the global spec computed features)

    # dataset experiments
    "partition_id_level": 0.,
    "percentage_ids": 0.
}
