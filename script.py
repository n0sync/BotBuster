import os
import re
import pandas as pd
from dotenv import load_dotenv
from data.data import TMDBMovieFetcher
from model.model import MovieRecommender

async def run_recommender(new_titles: list[str], top_n: int = 3):
    load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return ["Error: TMDB_API_KEY not found in .env file!"]

    fetcher = TMDBMovieFetcher(api_key)
    movie_titles = fetcher.read_txt_files(".")
    movie_titles = list(set(movie_titles))

    movies_data = await fetcher.fetch_all_movies(movie_titles)
    if not movies_data:
        return ["Error: No movie data was fetched!"]

    fetcher.save_to_csv(movies_data)

    recommender = MovieRecommender(catalog_path="data/movies.csv", user_path="watchlist.csv")
    recommender.fit()
    recs = recommender.recommend(top_n=top_n)

    results = []
    seen = set()
    for _, row in recs.iterrows():
        title = str(row.get("title", "")).strip()
        if not title or title in seen:
            continue
        seen.add(title)
        reason = str(recommender.explain_recommendation(row, recommender.user))
        overview = str(row.get("overview", ""))
        if len(overview) > 200:
            overview = overview[:200] + "..."
        results.append(f"**{title}**\nWhy watch: {reason}\nOverview: {overview}\n")

    return results if results else ["No recommendations could be generated."]
