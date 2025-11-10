import pandas as pd
import numpy as np
from typing import List
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, catalog_path="data/movies.csv", user_path="watchlist.csv"):
        self.catalog_path = catalog_path
        self.user_path = user_path
        self.catalog = None
        self.user = None
        self.vectorizers = {}
        self.catalog_matrix = None
        self.user_matrix = None
        self.user_profile = None
        self.scaler = MinMaxScaler()
        self.text_cols = ["genres", "keywords", "cast", "director", "overview"]
        self.num_cols = ["vote_average", "popularity", "vote_count"]
        self.id_col = "id"
        self.title_col = "title"

    def load(self):
        self.catalog = pd.read_csv(self.catalog_path)
        self.user = pd.read_csv(self.user_path)
        for c in self.text_cols:
            if c not in self.catalog.columns:
                self.catalog[c] = ""
            if c not in self.user.columns:
                self.user[c] = ""
            self.catalog[c] = self.catalog[c].astype(str).fillna("")
            self.user[c] = self.user[c].astype(str).fillna("")
        for c in self.num_cols:
            if c not in self.catalog.columns:
                self.catalog[c] = 0.0
            if c not in self.user.columns:
                self.user[c] = 0.0
            self.catalog[c] = pd.to_numeric(self.catalog[c], errors="coerce").fillna(0.0)
            self.user[c] = pd.to_numeric(self.user[c], errors="coerce").fillna(0.0)
        if self.id_col not in self.catalog.columns:
            self.catalog[self.id_col] = np.arange(len(self.catalog))
        if self.id_col not in self.user.columns:
            self.user[self.id_col] = np.arange(len(self.user)) + 10_000
        if self.title_col not in self.catalog.columns:
            self.catalog[self.title_col] = ""
        if self.title_col not in self.user.columns:
            self.user[self.title_col] = ""
        self.catalog = self.catalog[
            (self.catalog["vote_average"] >= 6.5) &
            (self.catalog["vote_count"] >= 500) &
            (self.catalog["popularity"] >= self.catalog["popularity"].quantile(0.2)) &
            (self.catalog["overview"].str.len() > 50)
        ]

    def fit(self, max_features: int = 20000):
        self.load()
        weights = {"genres":0.3,"keywords":0.2,"overview":0.3,"cast":0.1,"director":0.1}
        catalog_matrices = []
        user_matrices = []
        for col, w in weights.items():
            vec = TfidfVectorizer(stop_words="english", max_features=max_features)
            all_text = pd.concat([self.catalog[col], self.user[col]], ignore_index=True)
            vec.fit(all_text)
            self.vectorizers[col] = vec
            catalog_matrices.append(vec.transform(self.catalog[col]) * w)
            user_matrices.append(vec.transform(self.user[col]) * w)
        self.catalog_matrix = hstack(catalog_matrices)
        self.user_matrix = hstack(user_matrices)
        self.user_profile = np.asarray(self.user_matrix.mean(axis=0)).ravel()
        self.catalog["num_features"] = self.scaler.fit_transform(
            self.catalog[["vote_average","popularity"]].values
        ).mean(axis=1)

    def recommend(self, top_n: int = 20, exclude_liked: bool = True, diversity: float = 0.0) -> pd.DataFrame:
        sims = cosine_similarity(self.user_profile.reshape(1, -1), self.catalog_matrix).flatten()
        self.catalog["content_score"] = sims
        self.catalog["hybrid_score"] = 0.9 * self.catalog["content_score"] + 0.1 * self.catalog["num_features"]
        if exclude_liked:
            liked_ids = set(self.user[self.id_col].tolist())
            liked_titles = set(self.user[self.title_col].tolist())
            mask = (~self.catalog[self.id_col].isin(liked_ids)) & (~self.catalog[self.title_col].isin(liked_titles))
            ranked = self.catalog[mask].copy()
        else:
            ranked = self.catalog.copy()
        ranked = ranked.sort_values("hybrid_score", ascending=False)
        if diversity > 0:
            out = []
            seen_genres = set()
            for _, row in ranked.iterrows():
                g = tuple(sorted(str(row["genres"]).split()))
                if len(out) >= top_n:
                    break
                if np.random.rand() < diversity or g not in seen_genres:
                    out.append(row)
                    seen_genres.add(g)
            result = pd.DataFrame(out)
        else:
            result = ranked.head(top_n)
        cols = [self.id_col, self.title_col, "genres", "keywords", "director",
                "cast", "overview", "vote_average", "popularity",
                "content_score", "hybrid_score"]
        return result[cols]

    def explain_recommendation(self, movie_row, user_movies):
        reasons = []
        user_genres = set(" ".join(user_movies["genres"].astype(str)).split())
        user_keywords = set(" ".join(user_movies["keywords"].astype(str)).split())
        user_cast = set(" ".join(user_movies["cast"].astype(str)).split())
        user_directors = set(user_movies["director"].astype(str))
        movie_genres = set(str(movie_row["genres"]).split())
        movie_keywords = set(str(movie_row["keywords"]).split())
        movie_cast = set(str(movie_row["cast"]).split())
        movie_director = str(movie_row["director"])
        if movie_genres & user_genres:
            reasons.append(f"shares genres: {', '.join(sorted(movie_genres & user_genres))}")
        if movie_keywords & user_keywords:
            reasons.append(f"explores themes: {', '.join(sorted(list(movie_keywords & user_keywords)[:2]))}")
        if movie_cast & user_cast:
            reasons.append(f"features actors you liked: {', '.join(sorted(list(movie_cast & user_cast)[:2]))}")
        if movie_director in user_directors:
            reasons.append(f"directed by {movie_director}, also in your favorites")
        if not reasons:
            reasons.append("shares narrative style with your watchlist")
        return " | ".join(reasons)

    def explain(self, k: int = 20) -> pd.DataFrame:
        vocab = np.array(self.vectorizers["overview"].get_feature_names_out())
        user_vec = np.asarray(self.user_profile).flatten()
        top_idx = user_vec.argsort()[::-1][:k]
        tokens = vocab[top_idx]
        weights = user_vec[top_idx]
        df = pd.DataFrame({"token": tokens, "weight": weights})
        return df.sort_values("weight", ascending=False)
