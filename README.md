# 🎌 AniLens – Anime Recommendation System

AniLens is a **content-based anime recommendation system** that suggests similar anime based on plot descriptions and metadata. It leverages **Natural Language Processing (NLP)** techniques — specifically **TF-IDF vectorization** and **cosine similarity** — to recommend anime that are close in theme, genre, and narrative style.


---

## 🚀 Features

* 🔍 **Search anime by title**
* 🤖 **Content-based recommendations** using TF-IDF
* 📊 Similarity scores displayed as percentages
* 🖼️ Anime posters with automatic fallback images
* 🚫 Adult/Hentai content filtering
* 💾 Precomputed ML artifacts for fast inference
* 🌐 Clean Flask + Jinja2 web interface

---

## 🏗️ System Architecture

```text
┌────────────────────┐
│   dataset/anime.csv│
└─────────┬──────────┘
          │
          ▼
┌────────────────────────────┐
│ tfidf_recommender.ipynb    │
│ (Data cleaning + NLP)      │
└─────────┬──────────────────┘
          │
          ▼
┌──────────────────────────────────────────┐
│ Offline ML Artifacts (.pkl files)         │
│  • anilens_data_fixed.pkl                 │
│  • anilens_similarity.pkl                 │
│  • anilens_vectorizer.pkl                 │
└─────────┬────────────────────────────────┘
          │ (Loaded at runtime)
          ▼
┌────────────────────────────┐
│ Flask Backend (app.py)     │
│ - Search                   │
│ - Recommendation logic     │
└─────────┬──────────────────┘
          │
          ▼
┌────────────────────────────┐
│ Jinja2 Templates            │
│ index.html / detail.html   │
└─────────┬──────────────────┘
          │
          ▼
┌────────────────────────────┐
│ User Browser (UI)          │
│ Search & Recommendations  │
└────────────────────────────┘
```

**Architecture Highlights:**

* Model training is done **offline** in Jupyter Notebook
* Flask only performs **inference**, ensuring fast responses
* Precomputed `.pkl` files avoid retraining during runtime
* Clear separation between **data, ML, backend, and frontend**

---

## 🧠 Machine Learning Approach

### Model Type

* Content-Based Recommender System

### Techniques Used

* **TF-IDF (Term Frequency–Inverse Document Frequency)**
* **Cosine Similarity**

### Features Used

* Anime synopsis (textual description)
* Genres (weighted more heavily)
* Type (TV, Movie, OVA, etc.)

### Why TF-IDF?

* Interpretable and easy to explain
* Works well for text-heavy data
* Efficient for small-to-medium datasets
* Suitable for real-time recommendation systems

---

## 🗂️ Project Structure

```text
.
├── tfidf_recommender.ipynb        # Model training & experimentation
├── app.py                         # Flask web application
├── fix_images.py                  # Data cleaning & TF-IDF similarity builder
├── dataset/
│   └── anime.csv                  # Raw anime dataset
├── templates/
│   ├── index.html                 # Search page
│   └── detail.html                # Anime detail & recommendations
├── anilens_data.pkl               # Intermediate processed dataset
├── anilens_data_fixed.pkl         # Final cleaned dataset used by Flask
├── anilens_similarity.pkl         # Cosine similarity matrix
├── anilens_vectorizer.pkl         # TF-IDF vectorizer
└── README.md
```

> ℹ️ The `.pkl` files are intentionally included so the application can run immediately without retraining the model.

---

## 🧪 Jupyter Notebook (`tfidf_recommender.ipynb`)

The notebook contains the full ML pipeline:

1. Loading and exploring the raw dataset
2. Cleaning anime synopses and genres
3. Filtering adult (Hentai) content
4. Feature engineering using a weighted textual representation ("anime DNA")
5. TF-IDF vectorization (unigrams + bigrams)
6. Cosine similarity computation
7. Saving trained artifacts for inference

---

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/anilens.git
cd anilens
```

### 2️⃣ Install Dependencies

```bash
pip install flask pandas numpy scikit-learn
```

### 3️⃣ Run the Flask App

```bash
python app.py
```

Open your browser at:

```
http://127.0.0.1:5000/
```

---

## 📊 Dataset

* Source: Public anime metadata dataset (e.g., MyAnimeList-based)
* File: `dataset/anime.csv`

Key columns used:

* `title`
* `synopsis`
* `genres`
* `type`
* `score`
* `image_url`

---


---

## 👤 Author

**[Your Name]**
Machine Learning & Web Development Project

---

⭐ If you like this project, feel free to star the repository!

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
