# 🎌 AniLens – Anime Recommendation System

AniLens is a content-based anime recommendation system that suggests similar anime titles based on textual features such as descriptions and genres.  
It applies Natural Language Processing (NLP) techniques to compute similarity between anime and serves recommendations through a lightweight web application.

---

## 🚀 Features

- Anime search by title
- Content-based anime recommendations
- TF-IDF text vectorization
- Cosine similarity matching
- Fast recommendations using precomputed data
- Simple web interface powered by Flask

---

## 🏗️ System Architecture

```text
Dataset
   ↓
Jupyter Notebook
(Data preprocessing + TF-IDF + similarity computation)
   ↓
Serialized artifacts (.pkl)
   ↓
Flask Backend
   ↓
Web Interface

📁 Project Structure
.
├── tfidf_recommender.ipynb # Model training & experimentation
├── app.py # Flask web application
├── fix_images.py # Data cleaning & TF-IDF similarity builder
├── dataset/
│ └── anime.csv # Raw anime dataset
├── templates/
│ ├── index.html # Search page
│ └── detail.html # Anime detail & recommendations
├── anilens_data.pkl # Intermediate processed dataset
├── anilens_data_fixed.pkl # Final cleaned dataset used by Flask
├── anilens_similarity.pkl # Cosine similarity matrix
├── anilens_vectorizer.pkl # TF-IDF vectorizer
└── README.md


🧠 Machine Learning Approach

Textual features are extracted from anime metadata

TF-IDF is used to transform text into numerical vectors

Cosine similarity measures similarity between anime

Similarity scores are precomputed and stored for efficiency

▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/your-username/AniLens.git
cd AniLens

2. Install dependencies
pip install -r requirements.txt

3. Run the application
python app.py

4. Open in browser
http://127.0.0.1:5000

📦 Model Artifacts

The .pkl files included in this repository are required to run the application:

Processed dataset

TF-IDF vectorizer

Similarity matrix

They are intentionally committed to avoid retraining and ensure fast inference.

🛠️ Technologies Used

Python

Pandas

Scikit-learn

Flask

HTML / CSS

Jupyter Notebook

📌 Notes

This is a content-based recommendation system

No user profiles or collaborative filtering

---

## 👤 Author

**[Ahmed Hamza]**

Machine Learning & Web Development Project

---
Contact
For any inquiries or support, please reach out to:

Email: huangtian1810@gmail.com

GitHub: [AhmedHamza](https://github.com/Huang1810)

⭐ If you like this project, feel free to star the repository!
