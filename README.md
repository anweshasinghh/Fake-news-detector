🌟 Fake News Detector with AI Hint 🕵️‍♀️📰
This project is a fascinating demonstration of how Artificial Intelligence can be applied to the challenge of fake news detection! Built with Python 🐍 and the Flask web framework 🌐, it provides a user-friendly interface to analyze news content and source credibility.

Key AI-Powered Features:
AI-powered Fact-Checking ✔️❌:
The core of this feature uses a Logistic Regression model trained on the LIAR dataset 📚.
It processes news text using TF-IDF Vectorization to convert words into numerical features.
While it can provide a "Likely Real News" or "Likely Fake News" hint, it's important to remember this is a simplified model 🧠. Due to its foundational design and the nature of the training data (primarily political statements), it may sometimes lean towards classifying content as "fake," even when it's real. This is a common challenge in basic NLP models and highlights thecomplexity of accurate fact-checking!

Source Credibility Assessment trustworthiness:
This component offers an initial assessment of a news source's reliability.
Currently, it uses a hardcoded dictionary 🗂️ of example news domains (e.g., "politifact.com" as reliable, "breitbart.com" as unreliable).
It extracts the domain from a provided URL to check against this dictionary, indicating if a source is "Reliable," "Unreliable," or "Unknown." This provides a glimpse into how source trustworthiness can be integrated into such a system!
Project Technologies & Structure:
Python: The primary programming language. 🐍
Flask: Lightweight web framework for the user interface. 💡
Scikit-learn: For machine learning functionalities (TF-IDF, Logistic Regression). 📊
Joblib: For saving and loading trained models and data. 💾
Project Structure: Organized with src/ for code, data/ for datasets, models/ for trained AI artifacts, and templates/ for the web UI. 📁
Important Disclaimer:
This project serves as a proof-of-concept and an educational tool 🎓, demonstrating the potential of AI in fact-checking. It is not a substitute for professional fact-checking organizations or human verification 🧑‍⚖️. Real-world fake news detectionrequires far more advanced models, extensive and continuously updated datasets, and often real-time external data validation, which are beyond the scope of this starter project.


Feel free to explore the code and see how these AI hints come to life! ✨
