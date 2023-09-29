# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from pathlib import Path
from bs4 import BeautifulSoup
import requests
import os
import time


def download_file(url, save_path):
    """
    download file from get request and save it in save_path
    """
    with requests.get(url, stream=True) as r:
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=32768):
                fd.write(chunk)


def download_all_files(url, save_path, file_type="", exclude_type=False):
    """
    download all files in url that match file_type and save them in save_path
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # iterate through files in source page provided
    folder_request = requests.get(url)
    for file in BeautifulSoup(folder_request.text, "html.parser").find_all('a'):
        file_path = Path(save_path, file.text)
        if not os.path.exists(file_path):
            # download if it is not a folder and match desired extensions
            type_match = not file.text.endswith(file_type) if exclude_type else file.text.endswith(file_type)
            if not file.text.endswith("/") and type_match:
                print(url+file.text)
                start = time.time()
                download_file(url+file.text, file_path)
                print("The file has been downloaded in", round(time.time()-start, 4), "seconds")


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
    compressed_output_path = Path(output_path, "compressed")
    # download all files from daic-woz page
    download_all_files(database_url, compressed_output_path, "P.zip")
    download_all_files(database_url, output_path+"/labels", "P.zip", True)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
