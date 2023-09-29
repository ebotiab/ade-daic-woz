from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
import IPython.display as ipd

from src.hyperparameters import PARAMS
from src.features.utils_audio import show_spectrogram, show_waveform
from src.features.get_specs_features import load_spec_features


class Participant:
    """
    Represents a participant, there is only one participant per interview and vice versa
    """

    def __init__(self, participant_id, data_path="data"):
        self.participant_id = participant_id
        self.metadata_path = str(Path(data_path, "processed", "labels", "metadata.csv"))
        self.metadata = pd.read_csv(self.metadata_path, index_col="Participant_ID").loc[self.participant_id]
        self.label = self.metadata["PHQ8_Binary"]
        self.score = self.metadata["PHQ8_Score"]
        self.partition = self.metadata["partition"]
        self.gender = self.metadata["Gender"]


class Fragment:
    """
    Represents an interview fragment
    """

    def __init__(self, participant_id, audio_times=None, audio_frames=None, data_path="data", device="cuda"):
        """
        Args:
            participant_id (int): participant or interview identifier
            audio_times (tuple float pair): contains the start and stop times in the fragment audio
            audio_frames (tuple float pair): contains the start and stop audio frames in the fragment audio
            data_path (string): path where data is located
            device (string): type of device to perform torch operations
        """
        self.participant_id = participant_id
        self.participant = Participant(self.participant_id)
        self.audio_path = str(Path(data_path, "raw", "audio", f"{self.participant_id}_AUDIO.wav"))

        # load fragment audio extension in different units of measure
        self.stft_params = PARAMS["stft"]
        self.sr = PARAMS["sr"]
        if not audio_times and not audio_frames:
            raise Exception("At least one of 'audio_times' or 'audio_frames' must be specified")
        self.audio_times = audio_times
        if not self.audio_times:
            self.audio_times = [librosa.samples_to_time(t, sr=self.sr) for t in audio_frames]
        self.audio_frames = audio_frames
        if not self.audio_frames:
            self.audio_frames = [librosa.time_to_samples(t, sr=self.sr) for t in self.audio_times]
        self.spec_frames = [librosa.samples_to_frames(f, **self.stft_params) for f in self.audio_frames]

        # load user available device
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device != device:
            print(f"'cpu' as device has been set since '{device}' device is not available")

        # load required stft transformation parameters
        self.n_mels = PARAMS["n_mels"]
        self.to_dB = torchaudio.transforms.AmplitudeToDB().to(self.device)
        # load required spec features to compute spec normalization
        self.spec_features_path = str(Path(data_path, "processed", "spec_features.txt"))
        self.spec_features = load_spec_features(self.spec_features_path)

    def get_audio(self, frames=None, times=None):
        if not frames:  # convert time to frame if provided or set them to default otherwise
            frames = [librosa.time_to_samples(t, sr=PARAMS["sr"]) for t in times] if times else self.audio_frames
        if frames[0] < self.audio_frames[0] or frames[1] > self.audio_frames[1]:  # check inputs are not out the scope
            raise Exception(f"Out of the limits start: {self.audio_frames[0]} and end: {self.audio_frames[1]} frames")
        audio, sr = torchaudio.load(self.audio_path, frames[0], frames[1] - frames[0])
        return audio.to(self.device), sr

    def get_interactive_audio(self, frames=None, times=None):
        audio, sr = self.get_audio(frames, times)
        return ipd.Audio(audio, rate=sr)

    def show_waveform(self, frames=None, times=None, title="", save_path=None):
        audio, sr = self.get_audio(frames, times)
        plot = show_waveform(audio.cpu().numpy()[0], title, save_path)
        return plot, audio, sr

    def get_spectrogram(self, audio_frames=None, audio_times=None, spec_frames=None, use_mel=True,
                        normalize="globally", transpose=False):
        if not audio_frames and not audio_times:  # convert spec frames to audio frames
            audio_frames = self.audio_frames
            if spec_frames:
                audio_frames = [librosa.frames_to_samples(f, sr=PARAMS["sr"], **PARAMS["stft"]) for f in spec_frames]
        audio, _ = self.get_audio(audio_frames, audio_times)
        if use_mel:
            stft = torchaudio.transforms.MelSpectrogram(**self.stft_params, n_mels=self.n_mels).to(self.device)
        else:
            stft = torchaudio.transforms.Spectrogram(**self.stft_params).to(self.device)
        spec = self.to_dB(stft(audio))
        spec = spec.transpose(1, 2).flip([1, ]) if transpose else spec  # TODO: ask if flip is important
        if normalize:
            # normalize spectrogram with local or global features depending on selected normalization type
            if normalize == "globally":
                print(self.spec_features)
                specs_mean = torch.tensor(self.spec_features["global_mean"]).to(self.device)
                specs_var = torch.tensor(self.spec_features["global_var"]).to(self.device)
            else:
                specs_mean, specs_var = spec.mean(), spec.var()
            spec = (spec - specs_mean) / (torch.sqrt(specs_var) + 1e-14)
        return spec

    def show_spectrogram(self, audio_frames=None, audio_times=None, spec_frames=None, use_mel=True, normalize=True,
                         transpose=False, add_color_bar=True, add_title=False, png_path=None, plot=False):
        spec = self.get_spectrogram(audio_frames, audio_times, spec_frames, use_mel, normalize, transpose)
        spec = spec[0].cpu().numpy()
        p_id = self.participant_id if add_title else None
        plot = show_spectrogram(spec, "mel" if use_mel else "linear", add_color_bar, p_id, png_path, plot)
        return plot, spec


class Interview(Fragment):
    """
    Represents a fragment consisted in a complete interview
    """

    def __init__(self, p_id, data_path="data", device="cpu"):
        """
        Args:
            p_id (int): participant or interview identifier
            data_path (string): path where data is located
        """
        self.id = p_id
        self.metadata_path = str(Path(data_path, "processed", "labels", "metadata.csv"))
        self.participant = Participant(self.id, data_path)
        self.partition = self.participant.partition
        self.audio_file_path = str(Path(data_path, "raw", "audio",  f"{self.id}_AUDIO.wav"))
        self.audio_times = 0, librosa.get_duration(filename=self.audio_file_path)
        super().__init__(p_id, self.audio_times, None, data_path, device)
        self.transcript_file_path = str(Path(data_path, "processed", "transcripts", f"{self.id}_TRANSCRIPT.csv"))

    def get_transcript(self, times=None):
        transcript = pd.read_csv(self.transcript_file_path)
        times = [self.audio_times[0] + t for t in times] if times else self.audio_times  # map to the correct audio sec
        if times[0] < self.audio_times[0] or times[1] > self.audio_times[1]:  # check inputs are not out the scope
            raise Exception(f"Out of the limits start: {self.audio_times[0]} and end: {self.audio_times[1]} frames")
        transcript = transcript.loc[(times[0] <= transcript["start_time"]) & (transcript["stop_time"] <= times[1])]
        return transcript.reset_index(drop=True)


class Segment(Fragment):
    """
    Represents a fragment consisted in a segment from an interview
    """

    def __init__(self, interview_id, segment_id, data_path="data", device="cuda"):
        """
        Args:
            interview_id (int): participant identifier
            segment_id (int): segment identifier
        """
        self.id = segment_id
        self.interview = Interview(interview_id, data_path, device)
        self.segment_data = self.interview.get_transcript().loc[self.id]
        segment_times = self.segment_data["start_time"].item(), self.segment_data["stop_time"].item()
        super().__init__(interview_id, segment_times, None, data_path, device)


class Sample(Fragment):
    """
    Represents a fragment consisted in a sample from an interview segment
    """

    def __init__(self, interview_id, segment_id, sample_id, data_path="data", device="cuda"):
        """
        Args:
            interview_id (int): participant identifier
            segment_id (int): segment identifier
            sample_id (int): sample identifier
        """
        self.id = sample_id
        self.segment = Segment(interview_id, segment_id, data_path, device)
        # GET AUDIO TIMES
        # get segment audio frames
        segment_audio_start, segment_audio_end = self.segment.audio_frames[0], self.segment.audio_frames[1]
        # get the number of samples that is possible to extract from the segment
        segment_num_samples = self.segment.segment_data["num_samples"]
        # get sample hop and sample len measured in audio frames
        sample_hop, sample_len = PARAMS["sample_hop_audio_frames"], PARAMS["sample_n_audio_frames"]
        # get maximum possible audio start frame for the first sample of the segment
        sample0_max_audio_start = segment_audio_end - sample_hop * (segment_num_samples - 1) - sample_len
        # get the start audio frame for the first sample of the segment
        if PARAMS["random_seed"] and segment_audio_start != sample0_max_audio_start:
            # select it randomly from the possible range (useless if sample_hop=1)
            range_sample0_audio_starts = range(segment_audio_start, sample0_max_audio_start,
                                               PARAMS["stft"]["hop_length"])
            np.random.seed(PARAMS["random_seed"] + self.segment.segment_data.name)  # 1st sample randomly for each seg
            sample0_audio_start = np.random.choice(range_sample0_audio_starts)
        else:
            sample0_audio_start = segment_audio_start
        # start audio frame for the idx sample
        sample_audio_start = int(sample0_audio_start + sample_id*sample_hop)
        audio_frames = sample_audio_start, sample_audio_start + sample_len
        super().__init__(interview_id, None, audio_frames, data_path, device)
        self.interview = self.segment.interview
        self.label = self.interview.participant.label


