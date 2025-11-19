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
            "overview": 0.40,
            "keywords": 0.35,
            "director": 0.15,
            "genres": 0.07,
            "cast": 0.03
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
        self.catalog["num_score"] = 0.75 * num_scaled[:, 0] + 0.25 * num_scaled[:, 1]


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
            0.70 * self.catalog["content_score"] +
            0.30 * self.catalog["num_score"]
        )
        
        user_genre_tokens = " ".join(self.user["genres"]).lower().split()
        user_genre_freq = pd.Series(user_genre_tokens).value_counts(normalize=True)
        
        adult_preferred_genres = {"thriller", "mystery", "science", "fiction", "psychological", "drama", "horror", "noir", "crime"}
        hard_blacklist_genres = {"family", "animation", "comedy", "romance", "musical", "adventure", "documentary", "fantasy", "western"}
        
        def strict_tone_filter(row):
            keywords_lower = str(row.get("keywords", "")).lower()
            overview_lower = str(row.get("overview", "")).lower()
            genres_lower = str(row.get("genres", "")).lower()
            combined = keywords_lower + " " + overview_lower + " " + genres_lower
            
            catalog_genres = set(genres_lower.split())
            
            if catalog_genres & hard_blacklist_genres:
                return -999.0
            
            ya_scifi_markers = ["young adult", "ya", "teen", "teenager", "high school", 
                               "coming of age", "chosen one", "hero journey", "save the world"]
            whimsical_markers = ["whimsical", "magical", "enchanted", "fairy", "wizard", 
                               "heartwarming", "feel good", "uplifting", "cheerful", "joyful"]
            sentimental_markers = ["inspirational", "triumph", "overcome", "heartfelt", 
                                  "touching", "emotional journey", "life affirming"]
            family_markers = ["family", "kids", "children", "animated", "animation", "disney"]
            war_drama_markers = ["war", "wwii", "world war", "vietnam", "soldier", "military", 
                                "veteran", "battle", "combat", "based on true story", 
                                "true story", "real story", "biography"]
            
            penalty = 0.0
            
            for marker in ya_scifi_markers:
                if marker in combined:
                    penalty -= 50.0
            
            for marker in whimsical_markers:
                if marker in combined:
                    penalty -= 40.0
            
            for marker in sentimental_markers:
                if marker in combined:
                    penalty -= 35.0
            
            for marker in family_markers:
                if marker in combined:
                    penalty -= 60.0
            
            war_count = 0
            for marker in war_drama_markers:
                if marker in combined:
                    war_count += 1
            
            if war_count >= 2:
                penalty -= 45.0
            elif war_count == 1:
                penalty -= 15.0
            
            return penalty
        
        tone_filter_scores = self.catalog.apply(strict_tone_filter, axis=1)
        self.catalog["hybrid_score"] += tone_filter_scores
        
        user_keywords = set()
        for k in self.user["keywords"].astype(str):
            tokens = str(k).lower().split()
            user_keywords.update(t for t in tokens if t and t != "nan")
        
        psychological_adult = {"psychological", "psycho", "mind", "consciousness", "identity", 
                              "memory", "mental", "perception", "paranoia", "obsession", 
                              "madness", "insanity", "dissociation"}
        
        cerebral_adult = {"cerebral", "intellectual", "complex", "philosophical", "existential", 
                         "metaphysical", "abstract", "thought provoking", "morality", "ethics"}
        
        noir_dark_adult = {"noir", "dark", "gritty", "brutal", "violent", "bleak", "disturbing", 
                          "intense", "haunting", "pessimistic", "cynical", "nihilistic"}
        
        scifi_adult_hard = {"dystopian", "dystopia", "cyberpunk", "artificial intelligence", 
                           "consciousness", "simulation", "reality", "time paradox", "multiverse",
                           "posthuman", "transhumanism", "singularity"}
        
        scifi_space_adult = {"space exploration", "alien contact", "first contact", "cosmic horror",
                            "deep space", "interstellar"}
        
        thriller_adult = {"thriller", "suspense", "mystery", "conspiracy", "detective",
                         "investigation", "murder", "crime", "twist", "noir"}
        
        user_has_psychological = bool(user_keywords & psychological_adult)
        user_has_cerebral = bool(user_keywords & cerebral_adult)
        user_has_noir = bool(user_keywords & noir_dark_adult)
        user_has_hard_scifi = bool(user_keywords & scifi_adult_hard)
        user_has_space_scifi = bool(user_keywords & scifi_space_adult)
        user_has_thriller = bool(user_keywords & thriller_adult)
        
        def adult_tone_boost(row):
            movie_kw = set(str(row.get("keywords", "")).lower().split())
            movie_overview = str(row.get("overview", "")).lower()
            movie_kw.discard("")
            movie_kw.discard("nan")
            
            boost = 0.0
            
            if user_has_psychological and (movie_kw & psychological_adult):
                boost += 0.25
            
            if user_has_cerebral and (movie_kw & cerebral_adult):
                boost += 0.22
            
            if user_has_noir and (movie_kw & noir_dark_adult):
                boost += 0.20
            
            if user_has_hard_scifi and (movie_kw & scifi_adult_hard):
                boost += 0.25
            
            if user_has_space_scifi and (movie_kw & scifi_space_adult):
                boost += 0.15
            
            if user_has_thriller and (movie_kw & thriller_adult):
                boost += 0.18
            
            adult_marker_count = sum([
                bool(movie_kw & psychological_adult),
                bool(movie_kw & cerebral_adult),
                bool(movie_kw & noir_dark_adult),
                bool(movie_kw & scifi_adult_hard),
                bool(movie_kw & thriller_adult)
            ])
            
            if adult_marker_count >= 2:
                boost += 0.15
            
            return boost
        
        adult_boosts = self.catalog.apply(adult_tone_boost, axis=1)
        self.catalog["hybrid_score"] += adult_boosts
        
        def genre_alignment_score(row):
            catalog_genres = set(str(row["genres"]).lower().split())
            
            base_score = sum(user_genre_freq.get(g, 0) for g in catalog_genres)
            adult_match = len(catalog_genres & adult_preferred_genres) * 0.10
            
            return base_score + adult_match
        
        self.catalog["genre_alignment"] = self.catalog.apply(genre_alignment_score, axis=1)
        self.catalog["hybrid_score"] += 0.15 * self.catalog["genre_alignment"]
        
        user_directors = set(
            d.strip() for d in self.user["director"].dropna().str.lower() 
            if d and str(d).strip() and str(d).lower() != "nan"
        )
        
        def director_match_with_tone_gate(row):
            director = str(row.get("director", "")).lower().strip()
            
            if not director or director == "nan" or director not in user_directors:
                return 0.0
            
            movie_kw = set(str(row.get("keywords", "")).lower().split())
            movie_genres = set(str(row.get("genres", "")).lower().split())
            
            if movie_genres & hard_blacklist_genres:
                return 0.0
            
            has_adult_tone = bool(
                (movie_kw & psychological_adult) or
                (movie_kw & cerebral_adult) or
                (movie_kw & noir_dark_adult) or
                (movie_kw & scifi_adult_hard) or
                (movie_kw & thriller_adult)
            )
            
            if has_adult_tone:
                return 1.0
            else:
                return 0.0
        
        director_boost = self.catalog.apply(director_match_with_tone_gate, axis=1) * 0.10
        self.catalog["hybrid_score"] += director_boost
        
        if exclude_liked:
            liked_ids = set(self.user[self.id_col].tolist())
            liked_titles = set(self.user[self.title_col].astype(str).tolist())
            mask = (~self.catalog[self.id_col].isin(liked_ids)) & \
                   (~self.catalog[self.title_col].astype(str).isin(liked_titles))
            ranked = self.catalog[mask].copy()
        else:
            ranked = self.catalog.copy()
        
        ranked = ranked[ranked["hybrid_score"] > -10.0].copy()
        
        def final_tone_validation(row):
            keywords_lower = str(row.get("keywords", "")).lower()
            overview_lower = str(row.get("overview", "")).lower()
            genres_lower = str(row.get("genres", "")).lower()
            
            forbidden_patterns = [
                "young adult", "ya", "teen romance", "high school", "coming of age",
                "heartwarming", "feel good", "family friendly", "disney", "pixar",
                "whimsical", "magical", "enchanted", "fairy tale", "chosen one",
                "inspirational", "triumph", "heartfelt", "life affirming"
            ]
            
            for pattern in forbidden_patterns:
                if pattern in keywords_lower or pattern in overview_lower or pattern in genres_lower:
                    return False
            
            return True
        
        ranked = ranked[ranked.apply(final_tone_validation, axis=1)].copy()
        
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
        
        user_directors = {}
        for idx, row in user_movies.iterrows():
            director = str(row["director"]).lower().strip()
            title = str(row["title"]).strip()
            if director and director != "nan" and title:
                if director not in user_directors:
                    user_directors[director] = []
                user_directors[director].append(title)
        
        movie_genres = set(str(movie_row.get("genres", "")).lower().split())
        movie_keywords = set(str(movie_row.get("keywords", "")).lower().split())
        movie_cast = set(str(movie_row.get("cast", "")).lower().split())
        movie_director = str(movie_row.get("director", "")).lower().strip()
        
        movie_genres.discard("")
        movie_genres.discard("nan")
        movie_keywords.discard("")
        movie_keywords.discard("nan")
        movie_cast.discard("")
        movie_cast.discard("nan")
        
        psychological_keywords = {
            "psychological", "psycho", "mind", "consciousness", "identity", "memory",
            "mental", "perception", "reality", "paranoia", "obsession", "madness",
            "insanity", "dissociation", "schizophrenia", "delusion"
        }
        
        cerebral_keywords = {
            "cerebral", "intellectual", "complex", "philosophical", "existential",
            "metaphysical", "abstract", "thought", "provoking", "morality", "ethics"
        }
        
        noir_keywords = {
            "noir", "dark", "gritty", "brutal", "violent", "bleak", "disturbing",
            "intense", "haunting", "pessimistic", "cynical", "nihilistic"
        }
        
        scifi_adult_keywords = {
            "dystopian", "dystopia", "cyberpunk", "artificial", "intelligence",
            "consciousness", "simulation", "time", "paradox", "multiverse",
            "posthuman", "transhumanism", "singularity", "space", "alien"
        }
        
        thriller_keywords = {
            "thriller", "suspense", "mystery", "conspiracy", "detective",
            "investigation", "murder", "crime", "twist"
        }
        
        shared_psychological = (user_keywords & psychological_keywords) & (movie_keywords & psychological_keywords)
        shared_cerebral = (user_keywords & cerebral_keywords) & (movie_keywords & cerebral_keywords)
        shared_noir = (user_keywords & noir_keywords) & (movie_keywords & noir_keywords)
        shared_scifi = (user_keywords & scifi_adult_keywords) & (movie_keywords & scifi_adult_keywords)
        shared_thriller = (user_keywords & thriller_keywords) & (movie_keywords & thriller_keywords)
        
        genre_overlap = movie_genres & user_genres
        keyword_overlap = movie_keywords & user_keywords
        cast_overlap = movie_cast & user_cast
        
        director_titles = user_directors.get(movie_director, [])
        director_match = len(director_titles) > 0 and movie_director and movie_director != "nan"
        
        if director_match and director_titles:
            if len(director_titles) == 1:
                reasons.append(f"by {movie_director.title()}, who directed '{director_titles[0]}' in your watchlist")
            else:
                examples = director_titles[:2]
                reasons.append(f"by {movie_director.title()}, who directed {len(director_titles)} films in your watchlist including '{examples[0]}'")
        
        tone_matches = []
        if shared_psychological:
            sample = sorted(list(shared_psychological))[:2]
            tone_matches.append(f"psychological themes ({', '.join(sample)})")
        
        if shared_cerebral:
            sample = sorted(list(shared_cerebral))[:2]
            tone_matches.append(f"cerebral exploration ({', '.join(sample)})")
        
        if shared_noir:
            sample = sorted(list(shared_noir))[:2]
            tone_matches.append(f"dark/noir atmosphere ({', '.join(sample)})")
        
        if shared_scifi:
            sample = sorted(list(shared_scifi))[:2]
            tone_matches.append(f"adult sci-fi concepts ({', '.join(sample)})")
        
        if shared_thriller:
            sample = sorted(list(shared_thriller))[:2]
            tone_matches.append(f"thriller elements ({', '.join(sample)})")
        
        if tone_matches:
            if len(tone_matches) == 1:
                reasons.append(f"shares {tone_matches[0]}")
            elif len(tone_matches) == 2:
                reasons.append(f"combines {tone_matches[0]} with {tone_matches[1]}")
            else:
                reasons.append(f"merges {', '.join(tone_matches[:2])}, and {tone_matches[2]}")
        
        if genre_overlap and len(genre_overlap) > 0:
            genre_list = sorted(list(genre_overlap))[:2]
            if len(genre_list) == 1:
                reasons.append(f"genre match: {genre_list[0]}")
            else:
                reasons.append(f"genre overlap: {' and '.join(genre_list)}")
        
        if keyword_overlap and not tone_matches:
            all_tone_kw = psychological_keywords | cerebral_keywords | noir_keywords | scifi_adult_keywords | thriller_keywords
            thematic_only = keyword_overlap - all_tone_kw
            if thematic_only and len(thematic_only) > 0:
                sample = sorted(list(thematic_only))[:2]
                reasons.append(f"thematic overlap: {', '.join(sample)}")
        
        if cast_overlap and len(cast_overlap) >= 2:
            cast_sample = sorted(list(cast_overlap))[:2]
            reasons.append(f"features {' and '.join(cast_sample)}")
        
        content_score = float(movie_row.get("content_score", 0))
        
        if content_score >= 0.65 and len(reasons) == 0:
            reasons.append("strong narrative and thematic alignment with your taste")
        elif content_score >= 0.50 and len(reasons) == 0:
            reasons.append("similar storytelling approach to your favorites")
        elif content_score >= 0.35 and len(reasons) == 0:
            reasons.append("aligns with your viewing profile")
        
        if not reasons:
            reasons.append("selected based on compatibility with your preferences")
        
        return " | ".join(reasons)

    def explain(self, k: int = 20) -> pd.DataFrame:
        vocab = np.array(self.vectorizers["overview"].get_feature_names_out())
        user_vec = np.asarray(self.user_profile).flatten()
        top_idx = user_vec.argsort()[::-1][:k]
        tokens = vocab[top_idx]
        weights = user_vec[top_idx]
        df = pd.DataFrame({"token": tokens, "weight": weights})
        return df.sort_values("weight", ascending=False)