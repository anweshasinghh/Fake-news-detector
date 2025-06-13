import os
from flask import Flask, request, render_template
import joblib
import re

app = Flask(__name__)

# Load the trained models and data
MODEL_DIR = "models"

tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
fact_checking_model = joblib.load(os.path.join(MODEL_DIR, "fact_checking_model.joblib"))
source_credibility_dict = joblib.load(os.path.join(MODEL_DIR, "source_credibility_dict.joblib"))

# Define a threshold for classifying news as "Real" based on model's probability.
# The model provides probabilities for [fake, real]. A higher threshold here
# means it needs more confidence to classify as "Real News", thus increasing "Fake News" predictions.
# A lower threshold will make it classify more often as "Real News".
REAL_NEWS_THRESHOLD = 0.4 # Adjust this value to make the model more or less strict

def get_source_from_url(url):
    """Extracts the domain from a URL to use as a source."""
    if not url: # Handle empty URL string
        return None
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if match:
        return match.group(1)
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    fact_check_result = None
    source_credibility_result = None
    input_text = ""
    input_url = ""

    if request.method == "POST":
        input_text = request.form["news_text"]
        input_url = request.form["news_url"]

        # Fact-checking
        if input_text:
            text_vectorized = tfidf_vectorizer.transform([input_text])
            # Get prediction probabilities for each class (0: fake, 1: real)
            probabilities = fact_checking_model.predict_proba(text_vectorized)[0]
            proba_real = probabilities[1] # Probability of being real news

            if proba_real >= REAL_NEWS_THRESHOLD:
                fact_check_result = "Likely Real News"
            else:
                fact_check_result = "Likely Fake News"

        # Source credibility
        if input_url:
            source = get_source_from_url(input_url)
            if source:
                credibility = source_credibility_dict.get(source)
                if credibility is None:
                    source_credibility_result = f"Source credibility: Unknown for {source}"
                elif credibility == 1:
                    source_credibility_result = f"Source credibility: Reliable ({source})"
                else: # credibility == 0
                    source_credibility_result = f"Source credibility: Unreliable ({source})"
            else:
                source_credibility_result = "Source credibility: Invalid URL."
        else:
            source_credibility_result = "Source credibility: No URL provided."

    return render_template("index.html",
                           fact_check_result=fact_check_result,
                           source_credibility_result=source_credibility_result,
                           input_text=input_text,
                           input_url=input_url)

if __name__ == "__main__":
    app.run(debug=True) 