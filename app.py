from flask import Flask, render_template, request, url_for
import pickle
import pandas as pd
from urllib.parse import quote

app = Flask(__name__)

# ─── Load dataset & similarity ────────────────────────────────
print("Loading AniLens dataset...")
df = pd.read_pickle("anilens_data_fixed.pkl")

# You should have already precomputed similarity using your TF-IDF notebook:
similarity = pickle.load(open("anilens_similarity.pkl", "rb"))
print(f"→ Loaded {len(df)} anime")

# ─── Helper functions ────────────────────────────────────────
def search_anime(query):
    """Return all partial matches for a search query"""
    if not query:
        return []
    query = query.strip()
    matches = df[df['title'].str.contains(query, case=False, na=False)]
    return matches.to_dict('records')

def get_recommendations(mal_id, count=8):
    """Return a list of recommended anime based on similarity"""
    if mal_id not in df['mal_id'].values:
        return []
    idx = df[df['mal_id'] == mal_id].index[0]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    recs = []
    for i, score in scores[1:count+1]:
        row = df.iloc[i]
        genres = ', '.join(row.get('genres', [])[:6]) if isinstance(row.get('genres'), list) else row.get('genres', '—')
        recs.append({
            'mal_id': int(row['mal_id']),
            'title': row['title'],
            'similarity': round(score * 100, 1),
            'image': row.get('image_url', f"https://via.placeholder.com/250x360/140028/c084fc?text={quote(row['title'])}"),
            'genres': genres,
            'type': row.get('type', '—'),
            'score': round(row.get('score', 0), 1)
        })
    return recs

# ─── Routes ─────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def home():
    results = []
    error = None
    query = ""

    if request.method == 'POST':
        query = request.form.get('anime_title', '').strip()
        if query:
            results = search_anime(query)
            if not results:
                error = f"No anime found for '{query}'"

    return render_template('index.html', query=query, results=results, error=error)

@app.route('/anime/<int:mal_id>')
def anime_detail(mal_id):
    anime_row = df[df['mal_id'] == mal_id]
    if anime_row.empty:
        return render_template('error.html', message=f"Anime ID {mal_id} not found"), 404

    anime = anime_row.iloc[0].to_dict()
    recommendations = get_recommendations(mal_id)
    return render_template('detail.html', anime=anime, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
