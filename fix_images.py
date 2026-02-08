import pandas as pd
import numpy as np
import re
import pickle
from urllib.parse import quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── 1️⃣ Load CSV ───────────────────────────────
df = pd.read_csv("dataset/anime.csv", low_memory=False)

# ─── 2️⃣ Fix image URLs ─────────────────────────
def get_image_url(row):
    if pd.notna(row.get('image_jpg_large_url', None)):
        return row['image_jpg_large_url']
    if pd.notna(row.get('image_jpg_url', None)):
        return row['image_jpg_url']
    return f"https://via.placeholder.com/400x550/140028/c084fc?text={quote(row['title'])}"

df['image_url'] = df.apply(get_image_url, axis=1)

# ─── 3️⃣ Clean synopsis ─────────────────────────
def clean_synopsis(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)          # Remove HTML tags
    text = re.sub(r'[^a-z0-9\s\'-]', '', text)  # Keep letters, numbers, spaces, apostrophes, hyphens
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['synopsis_clean'] = df['synopsis'].apply(clean_synopsis)

# ─── 4️⃣ Parse genres ──────────────────────────
def parse_genres(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value]
    if isinstance(value, str):
        return [x.strip() for x in value.split(',') if x.strip()]
    return []

df['genres_clean'] = df['genres'].apply(parse_genres)

# ─── 5️⃣ Filter Hentai ─────────────────────────
df = df[~df['genres_clean'].apply(lambda g: 'Hentai' in g)].reset_index(drop=True)

# ─── 6️⃣ Create anime_dna with genres weighting ×5 ─────────────
df['anime_dna'] = (
    df['synopsis_clean'].str.split() +
    (df['genres_clean'] * 5)   # genres have extra weight
).apply(lambda x: " ".join(x))

# ─── 7️⃣ Compute TF-IDF similarity ─────────────
vectorizer = TfidfVectorizer(
    max_features=7000,
    stop_words='english',
    ngram_range=(1,2)
)
tfidf_matrix = vectorizer.fit_transform(df['anime_dna'])
similarity = cosine_similarity(tfidf_matrix)

# ─── 8️⃣ Keep only useful columns ─────────────
core_cols = ['mal_id', 'title', 'type', 'score', 'genres', 'synopsis', 'image_url']
final_data = df[core_cols].copy()

# ─── 9️⃣ Save dataset & similarity ────────────
final_data.to_pickle("anilens_data_fixed.pkl")
pickle.dump(similarity, open("anilens_similarity.pkl", "wb"))

print("✅ Dataset and similarity saved successfully!")
print(f"→ {len(final_data)} anime ready for Flask")