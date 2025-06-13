import os
import requests
import pandas as pd
import joblib

# URL for the NELA-GT-2020 labels.csv
labels_url = "https://raw.githubusercontent.com/MELALab/nela-gt/main/data/labels.csv"

# Directory to save the data
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

def download_source_labels():
    """Downloads the NELA-GT-2020 source labels CSV file."""
    print("Downloading NELA-GT-2020 source labels...")
    file_path = os.path.join(data_dir, "nela_gt_2020_labels.csv")
    if not os.path.exists(file_path):
        response = requests.get(labels_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded nela_gt_2020_labels.csv to {data_dir}/")
    else:
        print(f"nela_gt_2020_labels.csv already exists in {data_dir}/, skipping download.")

def load_source_credibility_data():
    """Loads the source credibility data into a pandas DataFrame."""
    file_path = os.path.join(data_dir, "nela_gt_2020_labels.csv")
    if not os.path.exists(file_path):
        download_source_labels()
    labels_df = pd.read_csv(file_path)
    print(f"Loaded nela_gt_2020_labels.csv with {len(labels_df)} rows.")
    return labels_df

def train_source_credibility_model():
    """Creates a hardcoded dictionary for source credibility and saves it."""
    print("Creating hardcoded source credibility dictionary...")
    # This is a simplified, hardcoded example for demonstration purposes.
    # In a real application, this would come from a more comprehensive dataset or API.
    source_credibility_dict = {
        "politifact.com": 1,  # Reliable
        "nytimes.com": 1,     # Reliable
        "bbc.com": 1,         # Reliable
        "breitbart.com": 0,   # Unreliable (example)
        "infowars.com": 0,    # Unreliable (example)
        "theonion.com": 0,    # Satire, treated as unreliable for factual news
    }

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(source_credibility_dict, os.path.join(models_dir, "source_credibility_dict.joblib"))
    print(f"Source credibility dictionary saved to {models_dir}/source_credibility_dict.joblib")

if __name__ == "__main__":
    train_source_credibility_model() 