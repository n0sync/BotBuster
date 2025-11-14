import pandas as pd
import numpy as np
import re
from typing import List, Optional
from collections import deque
from scipy.sparse import hstack, csr_matrix
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
        self.recent_history = deque(maxlen=100)
        self.random_seed: Optional[int] = None

    def _normalize_text(self, s):
        s = str(s).lower()
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _normalize_list(self, s):
        s = str(s)
        if not s or s.lower() == "nan":
            return ""
        parts = re.split(r"[|,;/]", s)
        parts = [self._normalize_text(p) for p in parts if p and p.lower() != "nan"]
        return " ".join(sorted(set(parts)))

    def _normalize_row(self, df):
        df["genres"] = df["genres"].apply(self._normalize_list)
        df["keywords"] = df["keywords"].apply(self._normalize_list)
        df["cast"] = df["cast"].apply(self._normalize_list)
        df["director"] = df["director"].apply(self._normalize_text)
        df["overview"] = df["overview"].astype(str).apply(self._normalize_text)
        return df

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
        self.catalog = self._normalize_row(self.catalog)
        self.user = self._normalize_row(self.user)
        self.catalog["overview_len"] = self.catalog["overview"].str.len()
        if len(self.catalog) > 0:
            self.catalog = self.catalog[
                (self.catalog["vote_average"] >= self.catalog["vote_average"].quantile(0.2)) &
                (self.catalog["vote_count"] >= self.catalog["vote_count"].quantile(0.2)) &
                (self.catalog["overview_len"] >= 30)
            ].copy()
        self.catalog.reset_index(drop=True, inplace=True)
    def fit(self, max_features: int = 30000):
        self.load()
        weights = {"genres":0.35,"keywords":0.3,"overview":0.25,"cast":0.1,"director":0.0}
        catalog_matrices = []
        user_matrices = []
        for col, w in weights.items():
            vec = TfidfVectorizer(stop_words="english",
                                max_features=max_features,
                                ngram_range=(1,2),
                                min_df=2)
            all_text = pd.concat([self.catalog[col], self.user[col]], ignore_index=True)
            vec.fit(all_text)

            self.vectorizers[col] = vec
            cm = vec.transform(self.catalog[col])
            um = vec.transform(self.user[col])

            catalog_matrices.append(cm * w)
            user_matrices.append(um * w)

        self.catalog_matrix = hstack(catalog_matrices).tocsr()
        self.user_matrix = hstack(user_matrices).tocsr()

        if self.user_matrix.shape[0] > 0:
            qa = self.user[["vote_average","popularity","vote_count"]].copy()
            qa_scaled = MinMaxScaler().fit_transform(qa.values)
            w = np.asarray(qa_scaled.mean(axis=1)).reshape(-1,1)
            wm = csr_matrix(w.T)
            try:
                self.user_profile = wm.dot(self.user_matrix).toarray().flatten()
                n = np.linalg.norm(self.user_profile)
                if n > 0:
                    self.user_profile = self.user_profile / n
            except Exception:
                self.user_profile = np.zeros(self.catalog_matrix.shape[1])
        else:
            self.user_profile = np.zeros(self.catalog_matrix.shape[1])

        if not self.catalog.empty:
            num_scaled = self.scaler.fit_transform(
                self.catalog[["vote_average","popularity","vote_count"]].values
            )
            self.catalog["num_score"] = num_scaled.mean(axis=1)
        else:
            self.catalog["num_score"] = 0.0


    def _mmr(self, candidates, top_n, lambda_div=0.5):
        selected = []
        candidate_indices = candidates.index.tolist()
        while len(selected) < top_n and candidate_indices:
            if not selected:
                best_idx = candidate_indices[0]
                selected.append(best_idx)
                candidate_indices.pop(0)
                continue
            cand_vecs = self.catalog_matrix[candidate_indices]
            sel_vecs = self.catalog_matrix[selected]
            sims = cosine_similarity(cand_vecs, sel_vecs).max(axis=1)
            relevance = candidates.loc[candidate_indices, "hybrid_score"].values
            mmr_score = lambda_div * relevance - (1 - lambda_div) * sims
            best_pos = mmr_score.argmax()
            best_idx = candidate_indices[best_pos]
            selected.append(best_idx)
            candidate_indices.pop(best_pos)
        return candidates.loc[selected]

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if temperature <= 0:
            temperature = 1e-6
        x = np.array(x, dtype=float)
        x = x - x.max()
        exp = np.exp(x / temperature)
        s = exp / (exp.sum() + 1e-12)
        return s

    def _stochastic_diverse_selection(self, candidates: pd.DataFrame, top_n: int,
                                      diversity: float = 0.5, temperature: float = 0.7,
                                      exclude_ids: Optional[set] = None) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        scores = candidates['hybrid_score'].fillna(0.0).values.astype(float)
        rng = np.random.RandomState(self.random_seed)
        if exclude_ids:
            mask_excl = candidates[self.id_col].isin(exclude_ids)
            scores = np.where(mask_excl, -np.inf, scores)

        selected_idx = []
        candidate_idx = list(range(len(candidates)))

        try:
            vecs = self.catalog_matrix[candidates.index]
            sim_matrix = cosine_similarity(vecs)
        except Exception:
            sim_matrix = None

        for _ in range(min(top_n, len(candidate_idx))):
            probs = self._softmax(scores, temperature)
            probs = np.where(np.isfinite(scores) & (scores > -np.inf), probs, 0.0)
            if probs.sum() <= 0:
                remaining = [i for i in candidate_idx if i not in selected_idx]
                remaining_scores = [(i, scores[i]) for i in remaining]
                remaining_scores.sort(key=lambda x: x[1], reverse=True)
                choice = remaining_scores[0][0]
            else:
                choice = rng.choice(len(probs), p=probs)
                tries = 0
                while (choice in selected_idx or scores[choice] == -np.inf) and tries < 20:
                    choice = rng.choice(len(probs), p=probs)
                    tries += 1
                if choice in selected_idx or scores[choice] == -np.inf:
                    remaining = [i for i in candidate_idx if i not in selected_idx and scores[i] > -np.inf]
                    if not remaining:
                        break
                    choice = max(remaining, key=lambda i: scores[i])

            selected_idx.append(choice)
            if sim_matrix is not None:
                sims = sim_matrix[choice]
                penalty = (1.0 - diversity) * sims
                scores = scores - penalty
            scores[choice] = -np.inf

        sel_positions = [candidates.index[i] for i in selected_idx]
        return candidates.loc[sel_positions]


    def recommend(self, top_n: int = 20, exclude_liked: bool = True, diversity: float = 0.5,
                  stochastic: bool = True, temperature: float = 0.7, seed: Optional[int] = None,
                  history_exclude: bool = True) -> pd.DataFrame:
        self.random_seed = seed

        sims = cosine_similarity(self.user_profile.reshape(1, -1), self.catalog_matrix).flatten()
        self.catalog["content_score"] = sims
        pop = self.catalog["popularity"].replace(0, np.nan).fillna(self.catalog["popularity"].median())
        novelty = 1.0 - (pop / (pop.max() + 1e-8))
        self.catalog["novelty_score"] = novelty
        self.catalog["hybrid_score"] = (
            0.6 * self.catalog["content_score"] +
            0.2 * self.catalog["num_score"] +
            0.2 * self.catalog["novelty_score"]
        )
        if exclude_liked:
            liked_ids = set(self.user[self.id_col].tolist())
            liked_titles = set(self.user[self.title_col].astype(str).tolist())
            mask = (~self.catalog[self.id_col].isin(liked_ids)) & (~self.catalog[self.title_col].astype(str).isin(liked_titles))
            ranked = self.catalog[mask].copy()
        else:
            ranked = self.catalog.copy()
        ranked = ranked.sort_values("hybrid_score", ascending=False)
        ranked = ranked.drop_duplicates(subset=["title"]).reset_index(drop=True)
        exclude_ids = set()
        if history_exclude and len(self.recent_history) > 0:
            exclude_ids.update(set(self.recent_history))

        if stochastic:
            candidate_pool = ranked.head(max(200, top_n * 10)).copy()
            result = self._stochastic_diverse_selection(candidate_pool, top_n,
                                                       diversity=diversity,
                                                       temperature=temperature,
                                                       exclude_ids=exclude_ids)
        else:
            if exclude_ids:
                ranked = ranked[~ranked[self.id_col].isin(exclude_ids)].copy()
            result = self._mmr(ranked, top_n, lambda_div=diversity)
        cols = [self.id_col, self.title_col, "genres", "keywords", "director",
                "cast", "overview", "vote_average", "popularity",
                "content_score", "num_score", "novelty_score", "hybrid_score"]
        try:
            for mid in result[self.id_col].tolist():
                self.recent_history.append(mid)
        except Exception:
            pass

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
            reasons.append(f"explores themes: {', '.join(sorted(list(movie_keywords & user_keywords)[:3]))}")
        if movie_cast & user_cast:
            reasons.append(f"features actors you liked: {', '.join(sorted(list(movie_cast & user_cast)[:3]))}")
        if movie_director in user_directors and movie_director:
            reasons.append(f"directed by {movie_director}, also in your favorites")
        if "novelty_score" in movie_row and movie_row["novelty_score"] > 0.6:
            reasons.append("novel but highly relevant")
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