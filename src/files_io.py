import warnings
from typing import Tuple

import pandas as pd
import requests
import tarfile
import os


def download_and_extract_raw_datasets():
    """
    Downloads the raw datasets and extracts them.
    """
    raw_files = [
        "Udacity_AZDIAS_052018.csv",
        "Udacity_CUSTOMERS_052018.csv",
        "Udacity_MAILOUT_052018_TEST.csv",
        "Udacity_MAILOUT_052018_TRAIN.csv",
    ]

    files_already_exists = True

    for file_name in raw_files:
        if not os.path.isfile(f"./data/{file_name}"):
            files_already_exists = False
            break

    if not files_already_exists:
        print("Downloading raw datasets...")

        url = "https://video.udacity-data.com/topher/2024/August/66b9ba05_arvato_data.tar/arvato_data.tar.gz"
        filename = url.split('/')[-1]

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.raw.read())
            print("Download completed.")
        else:
            print("Failed to download the file.")
            return

        print("Extracting the file...")
        try:
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(path=".")
            print("Extraction completed.")
        except Exception as e:
            print(f"Failed to extract the file: {e}")
        finally:
            os.remove(filename)
            print("Downloaded tar.gz file removed.")
    else:
        print("Raw Datasets already exists.")


def load_raw_population_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the raw population datasets.

    :return: - general population dataset
             - customer dataset
    """
    warnings.simplefilter(action="ignore", category=pd.errors.DtypeWarning)

    population = pd.read_csv("./data/Udacity_AZDIAS_052018.csv", sep=";")
    population.columns = population.columns.str.lower()

    customer = pd.read_csv("./data/Udacity_CUSTOMERS_052018.csv", sep=";")
    customer.columns = customer.columns.str.lower()

    warnings.resetwarnings()
    print("Population datasets loaded.")

    return population, customer