from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# ─── Load model artifacts (done once when app starts) ──────────────────────────
print("Loading AniLens model artifacts...")

try:
    df = pd.read_pickle("anilens_data.pkl")
    similarity = pickle.load(open("anilens_similarity.pkl", "rb"))
    # vectorizer = pickle.load(open("anilens_vectorizer.pkl", "rb"))  # optional
    print(f"→ Loaded {len(df)} anime entries")
except Exception as e:
    print("Error loading artifacts:", e)
    df = None
    similarity = None

# ─── Recommendation logic ──────────────────────────────────────────────────────

def get_recommendations(title, count=7):
    if df is None or similarity is None:
        return None, "Model not loaded. Check pickle files."

    title = title.strip()
    matches = df[df['title'].str.contains(title, case=False, na=False)]

    if matches.empty:
        return None, f"No anime found matching '{title}'"

    # Take the first matching title (most exact match)
    idx = matches.index[0]
    selected_title = df.loc[idx, 'title']

    # Get similarity scores
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recs = []
    for i, score in sim_scores[1:count+1]:
        row = df.iloc[i]
        recs.append({
            'title': row['title'],
            'similarity': round(score * 100, 1),
            'image': row.get('image_url', 'https://via.placeholder.com/240x340/1a0033/9f7aea?text=No+Poster'),
            'genres': ', '.join(row['genres'][:6]) if isinstance(row['genres'], list) else row.get('genres', '—'),
            'type': row.get('type', '—'),
            'score': round(row.get('score', 0), 1)
        })

    return selected_title, recs

# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def home():
    query = None
    selected = None
    recommendations = None
    error = None

    if request.method == 'POST':
        query = request.form.get('anime_title', '').strip()
        if query:
            selected, recommendations = get_recommendations(query)
            if isinstance(recommendations, str):
                error = recommendations
                recommendations = None

    return render_template(
        'index.html',
        query=query,
        selected=selected,
        recommendations=recommendations,
        error=error
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
