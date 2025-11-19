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
        
        weights = {
            "overview": 0.35,
            "keywords": 0.30,
            "director": 0.20,
            "genres": 0.10,
            "cast": 0.05
        }
        
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

        self.user_profile = self.user_matrix.mean(axis=0).A1
        n = np.linalg.norm(self.user_profile)
        if n > 0:
            self.user_profile = self.user_profile / n

        num_data = self.catalog[["vote_average", "vote_count"]].values
        num_scaled = self.scaler.fit_transform(num_data)
        self.catalog["num_score"] = 0.7 * num_scaled[:, 0] + 0.3 * num_scaled[:, 1]


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


    def recommend(self, top_n: int = 20, exclude_liked: bool = True, 
                  diversity: float = 0.2, stochastic: bool = False, 
                  temperature: float = 0.15, seed: Optional[int] = None,
                  history_exclude: bool = True) -> pd.DataFrame:
    
        self.random_seed = seed
    
        sims = cosine_similarity(self.user_profile.reshape(1, -1), self.catalog_matrix).flatten()
        self.catalog["content_score"] = sims
        
        self.catalog["hybrid_score"] = (
            0.80 * self.catalog["content_score"] +
            0.20 * self.catalog["num_score"]
        )
        
        user_directors = set(
            d.strip() for d in self.user["director"].dropna().str.lower() 
            if d and str(d).strip() and str(d).lower() != "nan"
        )
        
        def director_match(director):
            d = str(director).lower().strip()
            return 1.0 if (d and d != "nan" and d in user_directors) else 0.0
        
        director_boost = self.catalog["director"].apply(director_match) * 0.20
        self.catalog["hybrid_score"] += director_boost
        
        user_genre_tokens = " ".join(self.user["genres"]).lower().split()
        user_genre_freq = pd.Series(user_genre_tokens).value_counts(normalize=True)
        
        preferred_genres = {"thriller", "mystery", "science", "fiction", "psychological", "drama", "horror"}
        negative_genres = {"family", "animation", "comedy", "romance", "musical", "adventure"}
        
        def genre_alignment_score(row):
            catalog_genres = set(str(row["genres"]).lower().split())
            
            base_score = sum(user_genre_freq.get(g, 0) for g in catalog_genres)
            
            preferred_match = len(catalog_genres & preferred_genres) * 0.05
            negative_match = len(catalog_genres & negative_genres) * -0.15
            
            return base_score + preferred_match + negative_match
        
        self.catalog["genre_alignment"] = self.catalog.apply(genre_alignment_score, axis=1)
        self.catalog["hybrid_score"] += 0.15 * self.catalog["genre_alignment"]
        
        def tone_penalty(row):
            keywords_lower = str(row.get("keywords", "")).lower()
            overview_lower = str(row.get("overview", "")).lower()
            combined = keywords_lower + " " + overview_lower
            
            light_terms = ["heartwarming", "feel good", "uplifting", "cheerful", "joyful", 
                          "lighthearted", "fun", "playful", "whimsical"]
            family_terms = ["family", "kids", "children", "animated"]
            
            penalty = 0.0
            for term in light_terms:
                if term in combined:
                    penalty -= 0.08
            for term in family_terms:
                if term in combined:
                    penalty -= 0.12
            
            return penalty
        
        tone_penalties = self.catalog.apply(tone_penalty, axis=1)
        self.catalog["hybrid_score"] += tone_penalties
        
        if exclude_liked:
            liked_ids = set(self.user[self.id_col].tolist())
            liked_titles = set(self.user[self.title_col].astype(str).tolist())
            mask = (~self.catalog[self.id_col].isin(liked_ids)) & \
                   (~self.catalog[self.title_col].astype(str).isin(liked_titles))
            ranked = self.catalog[mask].copy()
        else:
            ranked = self.catalog.copy()
        
        ranked = ranked.sort_values("hybrid_score", ascending=False)
        ranked = ranked.drop_duplicates(subset=["title"]).reset_index(drop=True)
        
        exclude_ids = set()
        if history_exclude and len(self.recent_history) > 0:
            exclude_ids.update(set(self.recent_history))
        
        if stochastic:
            candidate_pool = ranked.head(max(100, top_n * 5)).copy()
            result = self._stochastic_diverse_selection(
                candidate_pool, top_n,
                diversity=diversity,
                temperature=max(temperature, 0.1),
                exclude_ids=exclude_ids
            )
        else:
            if exclude_ids:
                ranked = ranked[~ranked[self.id_col].isin(exclude_ids)].copy()
            result = self._mmr(ranked, top_n, lambda_div=0.8)
        cols = [self.id_col, self.title_col, "genres", "keywords", "director",
                "cast", "overview", "vote_average", "popularity",
                "content_score", "num_score", "hybrid_score"]
        try:
            for mid in result[self.id_col].tolist():
                self.recent_history.append(mid)
        except Exception:
            pass

        return result[cols]

    def explain_recommendation(self, movie_row, user_movies):
        reasons = []
        
        user_genres = set()
        for g in user_movies["genres"].astype(str):
            tokens = str(g).lower().split()
            user_genres.update(t for t in tokens if t and t != "nan")
        
        user_keywords = set()
        for k in user_movies["keywords"].astype(str):
            tokens = str(k).lower().split()
            user_keywords.update(t for t in tokens if t and t != "nan")
        
        user_cast = set()
        for c in user_movies["cast"].astype(str):
            tokens = str(c).lower().split()
            user_cast.update(t for t in tokens if t and t != "nan")
        
        user_directors = set()
        for d in user_movies["director"].astype(str):
            d_clean = str(d).lower().strip()
            if d_clean and d_clean != "nan":
                user_directors.add(d_clean)
        
        user_titles = set()
        for t in user_movies["title"].astype(str):
            t_clean = str(t).strip()
            if t_clean:
                user_titles.add(t_clean)
        
        movie_genres = set(str(movie_row.get("genres", "")).lower().split())
        movie_keywords = set(str(movie_row.get("keywords", "")).lower().split())
        movie_cast = set(str(movie_row.get("cast", "")).lower().split())
        movie_director = str(movie_row.get("director", "")).lower().strip()
        movie_overview = str(movie_row.get("overview", "")).lower()
        
        movie_genres.discard("")
        movie_genres.discard("nan")
        movie_keywords.discard("")
        movie_keywords.discard("nan")
        movie_cast.discard("")
        movie_cast.discard("nan")
        
        psychological_keywords = {
            "psychological", "psycho", "mind", "consciousness", "identity", "memory",
            "mental", "perception", "reality", "paranoia", "obsession", "madness"
        }
        
        scifi_keywords = {
            "sci", "fi", "science", "fiction", "dystopian", "dystopia", "futuristic",
            "future", "artificial", "intelligence", "space", "alien", "technology"
        }
        
        thriller_keywords = {
            "thriller", "suspense", "mystery", "conspiracy", "detective",
            "investigation", "murder", "crime", "twist"
        }
        
        dark_keywords = {
            "dark", "noir", "gritty", "brutal", "violent", "bleak",
            "disturbing", "intense", "haunting"
        }
        
        cerebral_keywords = {
            "cerebral", "intellectual", "complex", "philosophical",
            "existential", "metaphysical", "abstract", "thought", "provoking"
        }
        
        all_tone_keywords = (psychological_keywords | scifi_keywords | 
                            thriller_keywords | dark_keywords | cerebral_keywords)
        
        user_tone_matches = user_keywords & all_tone_keywords
        movie_tone_matches = movie_keywords & all_tone_keywords
        
        shared_psychological = (user_keywords & psychological_keywords) & (movie_keywords & psychological_keywords)
        shared_scifi = (user_keywords & scifi_keywords) & (movie_keywords & scifi_keywords)
        shared_thriller = (user_keywords & thriller_keywords) & (movie_keywords & thriller_keywords)
        shared_dark = (user_keywords & dark_keywords) & (movie_keywords & dark_keywords)
        shared_cerebral = (user_keywords & cerebral_keywords) & (movie_keywords & cerebral_keywords)
        
        genre_overlap = movie_genres & user_genres
        keyword_overlap = movie_keywords & user_keywords
        cast_overlap = movie_cast & user_cast
        
        director_match = (movie_director in user_directors and 
                         movie_director and 
                         movie_director != "nan" and 
                         len(movie_director) > 2)
        
        if director_match:
            director_titles = user_movies[user_movies["director"].str.lower().str.strip() == movie_director]["title"].tolist()
            if director_titles:
                example = director_titles[0] if len(director_titles) == 1 else f"{director_titles[0]} and others"
                reasons.append(f"same director as '{example}' ({movie_director.title()})")
        
        tone_descriptions = []
        if shared_psychological:
            tone_descriptions.append("psychological depth")
        if shared_scifi:
            tone_descriptions.append("sci-fi concepts")
        if shared_thriller:
            tone_descriptions.append("thriller suspense")
        if shared_dark:
            tone_descriptions.append("dark atmosphere")
        if shared_cerebral:
            tone_descriptions.append("cerebral storytelling")
        
        if tone_descriptions:
            if len(tone_descriptions) == 1:
                reasons.append(f"matches your preference for {tone_descriptions[0]}")
            elif len(tone_descriptions) == 2:
                reasons.append(f"combines {tone_descriptions[0]} with {tone_descriptions[1]}")
            else:
                reasons.append(f"blends {', '.join(tone_descriptions[:2])}, and {tone_descriptions[2]}")
        
        if keyword_overlap and not tone_descriptions:
            thematic_keywords = keyword_overlap - all_tone_keywords
            if thematic_keywords and len(thematic_keywords) > 0:
                kw_sample = sorted(list(thematic_keywords))[:2]
                if len(kw_sample) > 0:
                    reasons.append(f"explores similar themes ({', '.join(kw_sample)})")
        
        if genre_overlap and not tone_descriptions:
            genre_list = sorted(list(genre_overlap))
            if len(genre_list) == 1:
                reasons.append(f"shares your taste in {genre_list[0]}")
            elif len(genre_list) >= 2:
                reasons.append(f"combines {genre_list[0]} and {genre_list[1]} elements")
        
        if cast_overlap and len(cast_overlap) >= 2:
            cast_sample = sorted(list(cast_overlap))[:2]
            reasons.append(f"features actors from your watchlist ({', '.join(cast_sample)})")
        
        content_score = float(movie_row.get("content_score", 0))
        if content_score >= 0.6 and len(reasons) == 0:
            reasons.append("strong thematic and narrative alignment with your taste")
        elif content_score >= 0.4 and len(reasons) == 0:
            reasons.append("similar storytelling style to your favorites")
        elif content_score >= 0.3 and len(reasons) == 0:
            reasons.append("aligns with your viewing preferences")
        
        if not reasons:
            reasons.append("recommended based on your profile")
        
        return " | ".join(reasons)

    def explain(self, k: int = 20) -> pd.DataFrame:
        vocab = np.array(self.vectorizers["overview"].get_feature_names_out())
        user_vec = np.asarray(self.user_profile).flatten()
        top_idx = user_vec.argsort()[::-1][:k]
        tokens = vocab[top_idx]
        weights = user_vec[top_idx]
        df = pd.DataFrame({"token": tokens, "weight": weights})
        return df.sort_values("weight", ascending=False)