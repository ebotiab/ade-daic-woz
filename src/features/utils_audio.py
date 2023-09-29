# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import soundfile
from datetime import time
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import streamlit as st

from src.hyperparameters import PARAMS
"""
This script creates spectrogram matrices from wav files that can be passed to the CNN.
"""


def librosa_stft(audio, png_name='tmp.png', save_file=""):
    """
    A function that converts a wav file into a spectrogram represented by a \
    matrix where rows represent frequency bins, columns represent time, and \
    the values of the matrix represent the decibel intensity. A matrix of \
    this form can be passed as input to the CNN after undergoing normalization.
    """
    if type(audio) == str:  # load audio
        audio, _ = librosa.load(audio)
    spec = librosa.stft(audio)  # compute spectrogram
    spec = log_scale_spec(spec)  # scale logarithmically
    spec = librosa.power_to_db(spec)  # amplitude to decibel
    if save_file:
        show_spectrogram(spec, png_name)
    return spec


def add_extra_plt(title="", png_path="", plot=None):
    if title:
        plt.title(title)
    if png_path:
        plt.savefig(png_path)
    if plot:
        plt.show()


def show_waveform(audio, title="", png_path="", plot=None):
    """
    Plot and save a file if desired of audio waveform.
    """
    plot = plt.figure(figsize=(14, 5))
    librosa.display.waveshow(audio, sr=PARAMS["sr"], alpha=0.5)
    add_extra_plt(title, png_path, plot)
    return plot


def show_spectrogram(im_matrix, y_axis="linear", add_color_bar=True, title="", png_path=None, plot=False):
    """
    Plot and save a file if desired of spectrogram.
    """
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(im_matrix, x_axis='time', y_axis=y_axis, sr=PARAMS["sr"], **PARAMS["stft"])
    if add_color_bar:
        plt.colorbar(format="%+2.f")
    add_extra_plt(title, png_path, plot)
    return plt


def log_scale_spec(spec, factor=10.):
    """
    Scale frequency axis logarithmically.
    """
    freq_bins, time_bins = np.shape(spec)

    scale = np.linspace(0, 1, freq_bins) ** factor
    scale *= (freq_bins - 1) / max(scale)
    scale = np.unique(np.round(scale))
    scale = scale.astype(int)

    # create spectrogram with new freq bins
    new_spec = np.complex128(np.zeros([len(scale), time_bins]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            new_spec[i, :] = np.sum(spec[scale[i]:, :], axis=0)
        else:
            new_spec[i, :] = np.sum(spec[scale[i]:scale[i + 1], :], axis=0)
    return new_spec


def secs2time(total_seconds, return_str=False):
    minutes, seconds = divmod(total_seconds, 60)
    t = time(0, int(minutes), int(seconds))
    if return_str:
        return t.strftime("%M:%S")
    return t


def display_interactive_audio(audio, sr, is_tensor=False):
    if is_tensor:
        audio = audio.cpu().numpy()[0]  # from torch tensor to np array
    soundfile.write('interview_subset.wav', audio, sr, 'PCM_24')
    st.audio('interview_subset.wav', format='audio/wav')
    os.remove('interview_subset.wav')


@click.command()
@click.option("-p", '--segmented_files_path', type=click.Path(exists=True), default="data/interim",
              help="directory containing participant folders with segmented wav files")
def main(segmented_files_path):
    """
    Walks through wav files in dir_name and creates pngs of the spectrogram's.
    This is a visual representation of what is passed to the CNN before
    normalization, although a cropped matrix representation is actually passed.
    """
    for subdir, dirs, files in os.walk(segmented_files_path):
        for file in files:
            if file.endswith('.wav'):
                wav_file = os.path.join(subdir, file)
                png_path = subdir + '/' + file[:-4] + '.png'
                print('Processing ' + file + '...')
                librosa_stft(wav_file, png_name=png_path, save_png=True)  # TODO


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
