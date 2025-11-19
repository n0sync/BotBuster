import os
import re
import pandas as pd
import aiohttp
import asyncio
from pathlib import Path
from typing import List, Dict

class TMDBMovieFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {"accept": "application/json"}

    async def search_movie(self, session: aiohttp.ClientSession, title: str) -> Dict:
        search_url = f"{self.base_url}/search/movie"
        params = {"api_key": self.api_key, "query": title, "language": "en-US"}
        try:
            async with session.get(search_url, params=params) as response:
                data = await response.json()
                if data.get("results"):
                    movie_id = data["results"][0]["id"]
                    return await self.get_movie_details(session, movie_id)
        except Exception as e:
            print(f"Error searching for '{title}': {e}")
        return None

    async def get_movie_details(self, session: aiohttp.ClientSession, movie_id: int) -> Dict:
        details_url = f"{self.base_url}/movie/{movie_id}"
        params = {"api_key": self.api_key, "append_to_response": "credits,keywords,videos"}
        try:
            async with session.get(details_url, params=params) as response:
                data = await response.json()
                return {
                    "id": data.get("id"),
                    "title": data.get("title"),
                    "original_title": data.get("original_title"),
                    "overview": data.get("overview"),
                    "release_date": data.get("release_date"),
                    "runtime": data.get("runtime"),
                    "vote_average": data.get("vote_average"),
                    "vote_count": data.get("vote_count"),
                    "popularity": data.get("popularity"),
                    "genres": [g["name"] for g in data.get("genres", [])],
                    "production_companies": [c["name"] for c in data.get("production_companies", [])],
                    "production_countries": [c["name"] for c in data.get("production_countries", [])],
                    "spoken_languages": [l["english_name"] for l in data.get("spoken_languages", [])],
                    "budget": data.get("budget"),
                    "revenue": data.get("revenue"),
                    "tagline": data.get("tagline"),
                    "status": data.get("status"),
                    "director": self._get_director(data.get("credits", {})),
                    "cast": self._get_cast(data.get("credits", {}), limit=10),
                    "keywords": [k["name"] for k in data.get("keywords", {}).get("keywords", [])],
                    "poster_path": data.get("poster_path"),
                    "backdrop_path": data.get("backdrop_path"),
                }
        except Exception as e:
            print(f"Error getting details for movie ID {movie_id}: {e}")
        return None

    def _get_director(self, credits: Dict) -> str:
        for person in credits.get("crew", []):
            if person.get("job") == "Director":
                return person.get("name")
        return None

    def _get_cast(self, credits: Dict, limit: int = 10) -> List[str]:
        return [actor.get("name") for actor in credits.get("cast", [])[:limit]]

    def read_txt_files(self, directory: str = ".") -> List[str]:
        movie_titles = []
        watchlist_file = Path(directory) / "watchlist.txt"
        if watchlist_file.exists():
            try:
                with open(watchlist_file, "r", encoding="utf-8") as f:
                    titles = []
                    for line in f:
                        title = line.strip()
                        if not title:
                            continue
                        title = re.sub(r'^\d+[\.\)]\s*', '', title)
                        title = re.sub(r'\[.*?\]', '', title).strip()
                        titles.append(title)
                    movie_titles.extend(titles)
                    print(f"Read {len(titles)} titles from {watchlist_file.name}")
            except Exception as e:
                print(f"Error reading {watchlist_file}: {e}")
        else:
            print("No watchlist.txt file found.")
        return list(dict.fromkeys(movie_titles))

    async def fetch_all_movies(self, titles: List[str]) -> List[Dict]:
        if Path("watchlist.csv").exists():
            try:
                df = pd.read_csv("watchlist.csv")
                if not df.empty and len(df.columns) > 0:
                    print("✓ watchlist.csv already exists, skipping fetch.")
                    return df.to_dict(orient="records")
                else:
                    print("⚠ watchlist.csv is empty or invalid, regenerating...")
                    Path("watchlist.csv").unlink()
            except Exception as e:
                print(f"⚠ Error reading watchlist.csv ({e}), regenerating...")
                Path("watchlist.csv").unlink(missing_ok=True)
                
        movies_data = []
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector, headers=self.headers) as session:
            tasks = [self.search_movie(session, title) for title in titles]
            for coro in asyncio.as_completed(tasks):
                movie = await coro
                if movie:
                    movies_data.append(movie)
        return movies_data

    def save_to_csv(self, data: List[Dict], output_file: str = "watchlist.csv"):
        try:
            df = pd.DataFrame(data)
            for col in ["genres", "keywords", "cast", "production_companies", "production_countries", "spoken_languages"]:
                df[col] = df[col].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
            df.to_csv(output_file, index=False, encoding="utf-8", quoting=1)
            print(f"\n✓ Successfully saved {len(data)} movies to {output_file}")
        except Exception as e:
            print(f"Error saving to CSV: {e}") 
