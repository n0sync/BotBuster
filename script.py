import os
from dotenv import load_dotenv
from data.data import TMDBMovieFetcher
from model.model import MovieRecommender

async def run_recommender(new_titles: list[str], top_n: int = 3):
    load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return ["Error: TMDB_API_KEY not found in .env file!"]

    fetcher = TMDBMovieFetcher(api_key)
    movie_titles = set(fetcher.read_txt_files("."))
    movie_titles.update(new_titles)
    movie_titles = list(movie_titles)

    movies_data = await fetcher.fetch_all_movies(movie_titles)
    if not movies_data:
        return ["Error: No movie data was fetched!"]

    fetcher.save_to_csv(movies_data)

    recommender = MovieRecommender(
        catalog_path="data/movies.csv",
        user_path="watchlist.csv"
    )
    seed_env = os.getenv("RECOMMENDER_SEED")
    seed = int(seed_env) if seed_env and seed_env.isdigit() else None
    temp_env = os.getenv("RECOMMENDER_TEMPERATURE")
    try:
        temperature = float(temp_env) if temp_env is not None else 0.7
    except Exception:
        temperature = 0.7
    stochastic_env = os.getenv("RECOMMENDER_STOCHASTIC", "true").lower()
    stochastic = stochastic_env in ("1", "true", "yes", "y")
    history_env = os.getenv("RECOMMENDER_HISTORY_EXCLUDE", "true").lower()
    history_exclude = history_env in ("1", "true", "yes", "y")
    try:
        recommender.fit()
        recs = recommender.recommend(top_n=top_n, diversity=0.5, stochastic=stochastic, temperature=temperature, seed=seed, history_exclude=history_exclude)
    except Exception as e:
        return [f"Error generating recommendations: {e}"]

    results = []
    seen = set()
    for _, row in recs.iterrows():
        title = str(row.get("title", "")).strip()
        if not title or title in seen:
            continue
        seen.add(title)
        reason = recommender.explain_recommendation(row, recommender.user)
        overview = str(row.get("overview", "")).strip()
        if len(overview) > 200:
            overview = overview[:200].rsplit(" ", 1)[0] + "..."
        results.append(f"🎬 **{title}**\nWhy watch: {reason}\nOverview: {overview}\n")

    return results if results else ["No recommendations could be generated."]
