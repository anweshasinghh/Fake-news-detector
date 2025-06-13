import os
import requests
import pandas as pd

# URLs for the LIAR dataset files
urls = {
    "train": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/train.tsv",
    "test": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv",
    "valid": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/valid.tsv",
}

# Directory to save the data
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

def download_data():
    """Downloads the LIAR dataset files."""
    print("Downloading LIAR dataset...")
    for key, url in urls.items():
        file_path = os.path.join(data_dir, f"{key}.tsv")
        if not os.path.exists(file_path):
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {key}.tsv to {data_dir}/")
        else:
            print(f"{key}.tsv already exists in {data_dir}/, skipping download.")

def load_data():
    """Loads the LIAR dataset into pandas DataFrames."""
    data = {}
    column_names = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "job_title",
        "state_info",
        "party_affiliation",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context",
    ]
    for key in urls.keys():
        file_path = os.path.join(data_dir, f"{key}.tsv")
        data[key] = pd.read_csv(file_path, sep="\t", header=None, names=column_names)
        print(f"Loaded {key}.tsv with {len(data[key])} rows.")
    return data

if __name__ == "__main__":
    download_data()
    liar_data = load_data()
    # You can now work with liar_data['train'], liar_data['test'], liar_data['valid']
    print("\nFirst 5 rows of the training data:")
    print(liar_data["train"].head()) 