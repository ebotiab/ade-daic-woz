# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from pathlib import Path
import os
import zipfile
import tarfile


def extract_files(compressed_file, out_dir, is_tar=True, delete_compressed=False):
    """
    A function takes in a compressed file and extracts the .wav file and
    *transcript.csv files into separate folders in a user specified directory.
    Parameters
    ----------
    compressed_file : filepath
        path to the folder containing the DAIC-WOZ zip files
    out_dir : filepath
        path to the desired directory where audio and transcript folders
        will be created
    is_tar : bool
        If true, interprets compressed file as a tar, interprets compressed file as zip otherwise
    delete_compressed : bool
        If true, deletes the compressed file once relevant files are extracted
    Returns
    -------
    Two directories :
        audio : containing the extracted wav files
        transcripts : containing the extracted transcript csv files
    """
    # create audio directory
    audio_dir = os.path.join(out_dir, 'audio')
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # create transcripts directory
    transcripts_dir = os.path.join(out_dir, 'transcripts')
    if not os.path.exists(audio_dir):
        os.makedirs(transcripts_dir)

    print(compressed_file)
    if is_tar:  # create object from compressed file and extract member names
        compressed_ref = tarfile.open(compressed_file, 'r')
        file_names = compressed_ref.getnames()
    else:
        compressed_ref = zipfile.ZipFile(compressed_file)
        file_names = compressed_ref.namelist()

    for f in file_names:  # iterate through files in compressed file
        if f.endswith('.wav'):
            compressed_ref.extract(f, audio_dir)
        elif f.lower().endswith('transcript.csv'):
            compressed_ref.extract(f, transcripts_dir)
    compressed_ref.close()

    if delete_compressed:
        os.remove(compressed_file)


@click.command()
@click.option("-p", '--compressed_files_path', type=click.Path(exists=True), default="data/raw/compressed",
              help="folder where compressed files with DAIC-WOZ raw data are located")
@click.option('--remove-compressed/--save-compressed', " /-S", default=True,
              help="Remove compressed files with DAIC-WOZ raw data")
@click.option("-o", '--output_path', type=click.Path(exists=True), default="data/raw/",
              help="Output folder path")
def main(compressed_files_path, remove_compressed, output_path):
    """
    iterates through zip files downloaded and extracts the wav and transcript files.
    """
    for file in os.listdir(compressed_files_path):
        zip_file = os.path.join(compressed_files_path, file)
        extract_files(zip_file, output_path, is_tar=False, delete_compressed=remove_compressed)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
