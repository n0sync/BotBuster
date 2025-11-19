# BotBuster

BotBuster is a movie recommendation system that runs as a Discord bot. It fetches metadata from TMDB, processes it with a machine-learning model, and returns movie suggestions based on a user's watchlist.

## Project Structure

```
BotBuster/
│
├── src/
│   ├── main.py                 
│   ├── data/
│   │   ├── data.py             
│   │   └── movies.csv          
│   │
│   └── model/
│       └── model.py
│
├── bot.py
├── watchlist.csv                      
├── watchlist.txt               
├── requirements.txt
├── README.md
└── .env
```

## How It Works

1. The user provides movie titles through Discord slash commands, which are added to `watchlist.txt`.
2. `main.py` loads environment variables from `.env` (including the TMDB API key and optional model configuration parameters).
3. The system fetches detailed information for these titles from TMDB, including genres, keywords, cast, directors, and description as there are in `movies.csv` to match it.
4. The fetched movie data is converted and saved as `watchlist.csv` for processing.
5. All gathered movie metadata is also cached in `movies.csv` to speed up future requests.
6. The machine-learning model loads both `movies.csv` (the complete cached dataset) and `watchlist.csv` (the user's preferences).
7. By analyzing the watchlist, the model builds a profile of the user's unique taste and identifies similar movies from the cached dataset.
8. `main.py` handles all data processing and model execution, then feeds the formatted recommendations to `bot.py`.
9. The bot sends the personalized recommendations directly inside Discord.

## Installation & Usage

1. Create and activate a Python virtual environment:

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/MacOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file:

```env
TMDB_API_KEY=your_tmdb_api_key  # Get from: https://www.themoviedb.org/settings/api
DISCORD_TOKEN=your_discord_bot_token  # Get from: https://discord.com/developers/applications
```

4. Run the bot:

```bash
python bot.py
```

5. Test if the bot is running using `/ping` in Discord.

Once running, the bot listens for slash commands in your server.

## Commands

| Command | Description |
|---------|-------------|
| `/ping` | Check if the bot is online and responsive. |
| `/add_movie <title>` | Add a single movie to the watchlist. |
| `/add_list` | Paste multiple movie titles (line- or comma-separated). |
| `/watchlist` | Display all saved movies. |
| `/recommend` | Get one movie recommendation. |
| `/recommend_n <number>` | Get multiple recommendations. |
| `/clear_watchlist` | Erase all stored watchlist entries. |
| `/clear_messages` | Delete all messages in the current channel (permission required). |

## Future Plans

- **Genre-specific recommendations**: Add `/recommend <genre>` command to get recommendations filtered by specific genres (e.g., `/recommend action`, `/recommend comedy`)
- **Multi-user support**: Enable separate watchlists for different Discord users by migrating from CSV files to a database system
- **Expanded movie catalog**: Increase the dataset beyond the current ~10,000 movies for more diversity and better recommendations
- **Rating system**: Allow users to rate recommended movies to improve future suggestions
- **Streaming availability**: Integrate with streaming service APIs to show where movies are available to watch
- **Movie details command**: Add `/movie_info <title>` to fetch detailed information about any movie
- **Watchlist analytics**: Provide statistics and insights about the user's viewing preferences

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to open a Pull Request. All contributions, bug reports, and feature requests are appreciated!

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for more details.