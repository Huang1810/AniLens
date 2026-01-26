from flask import Flask, render_template, request
import pickle
import pandas as pd
from urllib.parse import quote

app = Flask(__name__)

# ─── Load dataset & similarity ────────────────────────────────
print("Loading AniLens dataset...")

df = pd.read_pickle("anilens_data_fixed.pkl")

# 🚫 Robust hentai filter (works for both list and string genres)
df = df[~df["genres"].apply(
    lambda g: ("Hentai" in g) if isinstance(g, str) else ("Hentai" in g if isinstance(g, list) else False)
)]
df = df.reset_index(drop=True)  # align with similarity

similarity = pickle.load(open("anilens_similarity.pkl", "rb"))

print(f"→ Loaded {len(df)} anime")
print("→ Similarity matrix shape:", similarity.shape)


# ─── Helper functions ────────────────────────────────────────
def search_anime(query):
    if not query:
        return []

    query = query.strip()
    matches = df[df["title"].str.contains(query, case=False, na=False)]
    return matches.to_dict("records")


def _format_row(row, score=0.0):
    genres = (
        ", ".join(row.get("genres", [])[:6])
        if isinstance(row.get("genres"), list)
        else row.get("genres", "—")
    )

    return {
        "mal_id": int(row["mal_id"]),
        "title": row["title"],
        "similarity": round(score * 100, 1),
        "image": row.get(
            "image_url",
            f"https://via.placeholder.com/250x360/140028/c084fc?text={quote(row['title'])}",
        ),
        "genres": genres,
        "type": row.get("type", "—"),
        "score": round(row.get("score", 0), 1),
    }


def get_recommendations(mal_id, count=8):
    # Find anime index
    match = df.index[df["mal_id"] == mal_id]
    if len(match) == 0:
        return []

    idx = match[0]
    recs = []

    # ─── 1️⃣ Similarity-based recommendations ────────────────
    if idx < similarity.shape[0]:
        scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)

        for i, score in scores:
            if i == idx:
                continue
            row = df.iloc[i]

            # 🚫 Skip hentai
            if (isinstance(row.get("genres"), list) and "Hentai" in row.get("genres")) or \
               (isinstance(row.get("genres"), str) and "Hentai" in row.get("genres")):
                continue

            if score <= 0:
                continue

            recs.append(_format_row(row, score))

            if len(recs) == count:
                return recs

    # ─── 2️⃣ Fallback: genre + top-rated ─────────────────────
    base_row = df.iloc[idx]
    base_genres = base_row.get("genres", [])

    fallback = df[df["mal_id"] != mal_id].copy()

    if isinstance(base_genres, list) and base_genres:
        fallback["genre_match"] = fallback["genres"].apply(
            lambda g: len(set(g) & set(base_genres)) if isinstance(g, list) else 0
        )
        fallback = fallback.sort_values(by=["genre_match", "score"], ascending=False)
    else:
        fallback = fallback.sort_values(by="score", ascending=False)

    for _, row in fallback.head(count).iterrows():
        # 🚫 Skip hentai again
        if (isinstance(row.get("genres"), list) and "Hentai" in row.get("genres")) or \
           (isinstance(row.get("genres"), str) and "Hentai" in row.get("genres")):
            continue

        recs.append(_format_row(row, 0.0))

    return recs


# ─── Routes ─────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    error = None
    query = ""

    if request.method == "POST":
        query = request.form.get("anime_title", "").strip()
        if query:
            results = search_anime(query)
            if not results:
                error = f"No anime found for '{query}'"

    return render_template("index.html", query=query, results=results, error=error)


@app.route("/anime/<int:mal_id>")
def anime_detail(mal_id):
    anime_row = df[df["mal_id"] == mal_id]

    if anime_row.empty:
        return (
            render_template("error.html", message=f"Anime ID {mal_id} not found"),
            404,
        )

    anime = anime_row.iloc[0].to_dict()
    recommendations = get_recommendations(mal_id)

    return render_template("detail.html", anime=anime, recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
