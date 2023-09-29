from pathlib import Path
import random
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
import torchaudio

from src.hyperparameters import PARAMS
from src.features.get_specs_features import load_spec_features
from src.models import custom_classes
from src.features.build_processed_transcripts import add_cum_sample_vars
from src.features import balance_train_data


class DaicWozDataset(Dataset):  # TODO: update also the custom classes to hop bal (and make to inherit actions)
    def __init__(self, metadata, normalize_locally=True, random_seed=None, data_path="data", device="cuda"):
        """
        Args:
            metadata (string or pandas df): path where metadata file has been saved or metadata pd dataframe
            normalize_locally (bool): normalize the spectrograms locally or globally
            random_seed (int or None): seed to shuffle segments from each interview and samples from each segment
            data_path (string): path where data is located
            device (string): type of device to perform torch operations
        """

        if type(metadata) == str:
            self.metadata_path = metadata
            # load partition metadata
            col_types = {"first_sample": int, "last_sample": int}
            self.metadata = pd.read_csv(self.metadata_path, index_col="Participant_ID", dtype=col_types)
        else:
            self.metadata_path = ""
            self.metadata = metadata

        # save project folder paths
        self.data_path = data_path
        self.audios_path = str(Path(self.data_path, "raw", "audio"))
        self.transcripts_path = str(Path(self.data_path, "processed", "transcripts"))
        self.spec_features_path = str(Path(self.data_path, "processed", "spec_features.txt"))

        # save normalization type and load spec features (used if global spec normalization)
        self.normalize_locally = normalize_locally
        self.spec_features = load_spec_features(self.spec_features_path)

        # save random seed
        self.random_seed = random_seed

        # load device
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device != device:
            print(f"'cpu' as device has been set since '{device}' device is not available")

        # load parameters and functions for computing stft
        self.stft_mel = PARAMS["stft_mel"]
        self.stft_params = PARAMS["stft"]
        if self.stft_mel:
            self.n_mels = PARAMS["n_mels"]
            self.stft = torchaudio.transforms.MelSpectrogram(**self.stft_params, n_mels=self.n_mels).to(self.device)
        else:
            self.n_mels = 0
            self.stft = torchaudio.transforms.Spectrogram(**self.stft_params).to(self.device)
        self.to_dB = torchaudio.transforms.AmplitudeToDB().to(self.device)
        self.transpose_spec = PARAMS["transpose_spec"]

        # load targets array
        self.targets = self._get_targets()

    def __len__(self):
        n_samples = int(self.metadata.iloc[-1]["last_sample"])
        return n_samples

    def __getitem__(self, idx):
        idx = int(idx)
        spec = self.get_spec(idx)
        label = self.get_label(idx)
        return spec, label, idx

    def _get_targets(self):
        targets_unflatten = self.metadata.apply(lambda x: [x["PHQ8_Binary"], ] * int(x["num_samples"]), axis=1)
        targets = [x for xs in targets_unflatten.values for x in xs]
        return np.array(targets)

    def _get_interview_data(self, idx):
        interview_location = (self.metadata["first_sample"] <= idx) & (idx < self.metadata["last_sample"])
        interview_data = self.metadata.loc[interview_location].squeeze()
        return interview_data

    def _get_segment_data(self, idx):
        # load interview transcript that corresponds with idx
        interview_data = self._get_interview_data(idx)
        seg_id = int(interview_data["first_sample"].item())
        seg_idx = idx - seg_id
        transcript = pd.read_csv(f'{self.transcripts_path}/{interview_data.name}_TRANSCRIPT.csv')
        # load data from segment that corresponds with idx
        segment_location = (transcript["first_sample"] <= seg_idx) & (seg_idx < transcript["last_sample"])
        segment_data = transcript.loc[segment_location].squeeze()
        return interview_data, segment_data, seg_idx

    def _get_sample_data(self, idx):
        # load segment that corresponds with idx
        interview_data, segment_data, idx_segment = self._get_segment_data(idx)
        sample_id = idx_segment - segment_data["first_sample"]
        return interview_data, segment_data, sample_id

    def get_fragment_ids(self, idx):
        interview_data, segment_data, sample_id = self._get_sample_data(int(idx))
        return interview_data.name, segment_data.name, sample_id

    def _get_start_audio_frame(self, idx):
        # get audio path
        interview_data, segment_data, sample_id = self._get_sample_data(int(idx))
        # get segment audio frames
        seg_audio_start = librosa.time_to_samples(segment_data["start_time"], sr=PARAMS["sr"])
        seg_audio_end = librosa.time_to_samples(segment_data["stop_time"], sr=PARAMS["sr"])
        # get the number of samples that is possible to extract from the segment
        segment_num_samples = segment_data["num_samples"]
        # get sample hop and sample len measured in audio frames
        sample_hop, sample_len = PARAMS["sample_hop_audio_frames"], PARAMS["sample_n_audio_frames"]
        if "sample_hop" in interview_data:  # get sample hop computed from sample hop balance if performed
            sample_hop = librosa.time_to_samples(interview_data["sample_hop"], sr=PARAMS["sr"])
        # get maximum possible audio start frame for the first sample of the segment
        sample0_max_audio_start = seg_audio_end - sample_hop * (segment_num_samples - 1) - sample_len
        # get the start audio frame for the first sample of the segment
        if self.random_seed and seg_audio_start != sample0_max_audio_start:
            # select it randomly from the possible range (useless if sample_hop=1)
            range_sample0_audio_starts = range(seg_audio_start, sample0_max_audio_start, PARAMS["stft"]["hop_length"])
            np.random.seed(self.random_seed + segment_data.name)  # first sample is chosen randomly for each segment
            sample0_audio_start = np.random.choice(range_sample0_audio_starts)
        else:
            sample0_audio_start = seg_audio_start
        # return start audio frame of the sample and participant id
        return int(sample0_audio_start + sample_id * sample_hop), interview_data.name

    def get_audio(self, idx):
        sample_audio_start, participant_id = self._get_start_audio_frame(idx)
        # load audio
        audio_file_path = str(Path(self.audios_path, f'{participant_id}_AUDIO.wav'))
        audio, sr = torchaudio.load(audio_file_path, sample_audio_start, PARAMS["sample_n_audio_frames"])
        return audio.to(self.device), sr

    def get_spec(self, idx):
        audio, _ = self.get_audio(int(idx))
        # compute spec from audio and transpose
        spec = self.to_dB(self.stft(audio))
        if self.transpose_spec:
            spec = spec.transpose(1, 2).flip([1, ])  # TODO: ask if flip could be important
        # normalize spectrogram with local or global features depending on hypermarket defined
        if self.normalize_locally:
            specs_mean, specs_var = spec.mean(), spec.var()
        else:
            specs_mean = torch.tensor(self.spec_features["global_mean"])
            specs_var = torch.tensor(self.spec_features["global_var"])
        spec = (spec - specs_mean) / (torch.sqrt(specs_var) + 1e-14)
        return spec

    def get_label(self, idx):
        interview_data = self._get_interview_data(int(idx))
        return interview_data["PHQ8_Binary"]

    def get_idx(self, interview_id, segment_id=0, sample_id=0):
        segment_idx = self._get_segment_idx(interview_id, segment_id)
        return int(segment_idx + sample_id)

    def _get_segment_idx(self, interview_id, segment_id=0):
        segment = pd.read_csv(Path(self.transcripts_path, f"{interview_id}_TRANSCRIPT.csv")).loc[segment_id]
        participant_idx = self._get_interview_idx(interview_id)
        return int(participant_idx + segment["first_sample"].item())

    def _get_interview_idx(self, interview_id):
        return int(self.metadata.loc[interview_id]["first_sample"].item())

    def get_class(self, idx):
        idx = int(idx)
        i_row = self.metadata.loc[(self.metadata["first_sample"] <= idx) & (idx < self.metadata["last_sample"])]
        return int(i_row["PHQ8_Binary"].item()), int(i_row.index.item())

    def get_interview(self, idx):
        interview_id, _, _ = self.get_fragment_ids(int(idx))
        interview = custom_classes.Interview(interview_id, self.data_path, self.device)
        return interview

    def get_segment(self, idx):
        interview_id, segment_id, _ = self.get_fragment_ids(int(idx))
        segment = custom_classes.Segment(interview_id, segment_id, self.data_path, self.device)
        return segment

    def get_sample(self, idx):
        interview_id, seg_id, sample_id = self.get_fragment_ids(int(idx))
        sample = custom_classes.Sample(interview_id, seg_id, sample_id, self.data_path, self.device)
        return sample


class DaicWozDatasetReport(DaicWozDataset):
    """
    Modified version from DaicWozDataset dataset to check that classes from each batch are balanced (for streamlit app)
    """

    def __getitem__(self, idx):
        label, interview_id = self.get_class(idx)
        return label, interview_id, idx


class DaicWozDatasetTwoParticipants(DaicWozDataset):
    """
    Reduced version from DaicWozDataset consisted on a given percentage of samples from 2 interviews with opposed labels
    """

    def __init__(self, metadata, perc_partitions, n_samples_class=None, r_seed=0, dep_p_id=None, norm_p_id=None,
                 normalize_locally=True, random_seed=None, data_path="data", device="cuda"):

        super().__init__(metadata, normalize_locally, random_seed, data_path, device)
        self.perc_partitions = perc_partitions
        self.r_seed = r_seed
        self.dep_p_id = self._check_id(dep_p_id, 1)
        if self.dep_p_id != dep_p_id:
            print(f"'dep_p_id' param has been set to {self.dep_p_id}")
        self.norm_p_id = self._check_id(norm_p_id, 0)
        if self.norm_p_id != norm_p_id:
            print(f"'norm_p_id' param has been set to {self.norm_p_id}")
        self.n_samples_class, self.chosen_sample_idx = self._get_indexes(self.dep_p_id, self.norm_p_id, n_samples_class)
        if self.n_samples_class != n_samples_class:
            print(f"'n_samples_class' param has been set to {self.n_samples_class}")  # TODO: param confusing

    def _check_id(self, p_id, correct_class):
        """
        return given id if it belongs to an interview with correct class, returns first id with correct class otherwise
        """
        correct_class_ids = self.metadata.loc[self.metadata["PHQ8_Binary"] == correct_class].index
        first_id_correct_class = int(correct_class_ids[0])
        return p_id if p_id in correct_class_ids else first_id_correct_class

    def _get_indexes(self, dep_p_id, norm_p_id, n_samples):
        """
        return nº of samples per class and the selected sample idx for the partition as a list
        the partition idx list is a random selection from an exactly balanced idx from chosen participants
        """
        dep_p_df, norm_p_df = self.metadata.loc[dep_p_id], self.metadata.loc[norm_p_id]
        dep_n_samples = dep_p_df["last_sample"] - dep_p_df["first_sample"]
        norm_n_samples = norm_p_df["last_sample"] - norm_p_df["first_sample"]
        n_samples = min(n_samples, dep_n_samples, norm_n_samples) if n_samples else min(dep_n_samples, norm_n_samples)
        random.seed(self.r_seed)
        dep_chosen_idx = random.sample(range(dep_p_df["first_sample"], dep_p_df["last_sample"]), n_samples)
        norm_chosen_idx = random.sample(range(norm_p_df["first_sample"], norm_p_df["last_sample"]), n_samples)
        chosen_sample_idx = dep_chosen_idx + norm_chosen_idx
        random.shuffle(chosen_sample_idx)
        partition_range = (np.array(self.perc_partitions) * len(chosen_sample_idx)).astype(int)
        return n_samples, chosen_sample_idx[partition_range[0]:partition_range[1]]

    def __getitem__(self, idx):
        spec = self.get_spec(self.chosen_sample_idx[idx])
        label = self.get_label(self.chosen_sample_idx[idx])
        return spec, label

    def __len__(self):
        return len(self.chosen_sample_idx)


class DaicWozTwoParticipantsReport(DaicWozDatasetTwoParticipants):
    """
    Modified version from DaicWozDatasetTwoParticipants to check that classes from each batch are balanced (streamlit)
    """

    def __getitem__(self, idx):
        label, participant_id = self.get_class(self.chosen_sample_idx[idx])
        return label, participant_id, idx


class DaicWozIdSubset(DaicWozDataset):
    """
    DaicWoZDataset version that allows to create two dataset partitions using the same participants,
    by giving the option of performing the sample search in segments in the inverse order
    """

    def __init__(self, metadata, reverse, subset_perc,
                 normalize_locally=True, random_seed=None, data_path="data", device="cuda"):
        super().__init__(metadata, normalize_locally, random_seed, data_path, device)
        self.reverse = reverse
        self.subset_perc = 1 - subset_perc if self.reverse else subset_perc
        self.metadata["num_samples"] = (self.metadata["num_samples"] * self.subset_perc).astype(int)
        old_num_rows = self.metadata.shape[0]
        self.metadata = self.metadata.iloc[0:int(PARAMS["percentage_ids"] * self.metadata.shape[0])]
        self.metadata = balance_train_data.class_undersampling(self.metadata, 1.)
        print(f"using {self.metadata.shape[0]} instead of {old_num_rows}")
        self.metadata = add_cum_sample_vars(self.metadata)

    def _get_segment_data(self, idx):  # to reverse order of segments
        # load interview transcript that corresponds with idx
        interview_data = self._get_interview_data(idx)
        seg_id = int(interview_data["first_sample"].item())
        seg_idx = idx - seg_id
        transcript = pd.read_csv(f'{self.transcripts_path}/{interview_data.name}_TRANSCRIPT.csv')
        transcript = add_cum_sample_vars(transcript.iloc[::-1]) if self.reverse else transcript  # reverse transcript
        # load data from segment that corresponds with idx
        segment_location = (transcript["first_sample"] <= seg_idx) & (seg_idx < transcript["last_sample"])
        segment_data = transcript.loc[segment_location].squeeze()
        return interview_data, segment_data, seg_idx

    def get_audio(self, idx):  # to reverse order of samples
        sample_audio_start, participant_id = self._get_start_audio_frame(idx)
        if self.reverse:
            _, segment_data, _ = self._get_segment_data(idx)
            seg_audio_end = librosa.time_to_samples(segment_data["stop_time"], sr=PARAMS["sr"])
            sample_audio_start = seg_audio_end - sample_audio_start - PARAMS["sample_n_audio_frames"]
        # load audio
        audio_file_path = str(Path(self.audios_path, f'{participant_id}_AUDIO.wav'))
        audio, sr = torchaudio.load(audio_file_path, sample_audio_start, PARAMS["sample_n_audio_frames"])
        return audio.to(self.device), sr
