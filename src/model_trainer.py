import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

from data_processor import load_data # Assuming data_processor is in the same directory

def train_fact_checking_model():
    print("Loading LIAR data...")
    liar_data = load_data()
    train_df = liar_data["train"]
    test_df = liar_data["test"]
    valid_df = liar_data["valid"]

    # Combine all data for training the vectorizer and for a larger training set
    all_statements = pd.concat([train_df["statement"], test_df["statement"], valid_df["statement"]], axis=0)

    # Define our simplified labels
    # 'true', 'mostly-true' -> REAL (1)
    # 'false', 'pants-fire', 'barely-true', 'half-true' -> FAKE (0)
    label_mapping = {
        "true": 1,
        "mostly-true": 1,
        "half-true": 0,  # We'll treat half-true as fake for a binary model initially
        "barely-true": 0,
        "false": 0,
        "pants-fire": 0,
    }

    train_df["binary_label"] = train_df["label"].map(label_mapping)
    test_df["binary_label"] = test_df["label"].map(label_mapping)
    valid_df["binary_label"] = valid_df["label"].map(label_mapping)

    # Drop rows where binary_label is NaN (if any original labels weren't mapped)
    train_df.dropna(subset=["binary_label"], inplace=True)
    test_df.dropna(subset=["binary_label"], inplace=True)
    valid_df.dropna(subset=["binary_label"], inplace=True)

    X_train = train_df["statement"]
    y_train = train_df["binary_label"]
    X_test = test_df["statement"]
    y_test = test_df["binary_label"]

    print("Training TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print("Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectorized, y_train)

    print("Evaluating Model...")
    y_pred = model.predict(X_test_vectorized)
    print(classification_report(y_test, y_pred))

    # Save the model and vectorizer
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(model, os.path.join(models_dir, "fact_checking_model.joblib"))
    print(f"Model and vectorizer saved to {models_dir}/")

if __name__ == "__main__":
    train_fact_checking_model() 