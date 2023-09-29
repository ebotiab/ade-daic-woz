# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from pathlib import Path
import time

import subprocess

# TODO: change to original daic-woz version


def runcmd(cmd, verbose=False):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    start = time.time()
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    print("1: ", time.time() - start)


def download_all_files(url, save_path):
    """
    download all files in url that match file_type and save them in save_path
    """
    runcmd(f'wget -P {save_path} -r -np -nH --cut-dirs=3 -R index.html {url}', verbose=True)


@click.command()
@click.option('-d', '--database_url', type=click.Path(), envvar="DATABASE_URL", prompt='Database url',
              help="url where DAIC-WOZ can be downloaded")
@click.option("-o", '--output_path', type=click.Path(exists=True), default="data/raw",
              help="Output folder path")
def main(database_url, output_path):
    """
    download extended DAIC-WOZ dataset with the provided url,
    """
    logger = logging.getLogger(__name__)
    logger.info('downloading raw data in compressed files')

    # download compressed raw data
    compressed_output_path = Path(output_path, "compressed")
    download_all_files(database_url, compressed_output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
