import os
import re
from dotenv import load_dotenv
import discord
from discord import app_commands
from discord.ext import commands
from script import run_recommender  
from flask import Flask
from threading import Thread
import logging

app = Flask('')

@app.route('/')
def home():
    return "alive!"

def run():
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=8080)

Thread(target=run).start()

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

WATCHLIST_FILE = "watchlist.txt"

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")

@bot.tree.command(name="ping", description="Replies with Pong!")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong!")

@bot.tree.command(name="recommend", description="Add a movie and get top 3 recommendations")
async def recommend(interaction: discord.Interaction, movie_title: str):
    await interaction.response.defer()
    results = await run_recommender([movie_title], top_n=3)
    combined = "\n\n".join(results)   
    await interaction.followup.send(combined)

@bot.tree.command(name="recommend_more", description="Get top 10 recommendations")
async def recommend_more(interaction: discord.Interaction, movie_title: str):
    await interaction.response.defer()
    results = await run_recommender([movie_title], top_n=10)  
    combined = "\n\n".join(results)
    await interaction.followup.send(combined)

@bot.tree.command(name="add_list", description="Paste multiple movie titles at once")
async def add_list(interaction: discord.Interaction, movies: str):
    await interaction.response.defer()
    raw_titles = re.split(r"\d+\.\s*|,|\n", movies)
    titles = []
    for t in raw_titles:
        cleaned = re.sub(r"\[.*?\]", "", t)
        cleaned = cleaned.strip(" :.-")
        if cleaned:
            titles.append(cleaned)
    if not titles:
        await interaction.followup.send("No valid movie titles found in your input.")
        return
    results = await run_recommender(titles, top_n=5)
    combined = "\n\n".join(results)
    await interaction.followup.send(
        f"Added {len(titles)} movies to your watchlist.\n\nHere are some recommendations:\n\n{combined}"
    )

@bot.tree.command(name="watchlist", description="Show your current watchlist")
async def watchlist(interaction: discord.Interaction):
    if not os.path.exists(WATCHLIST_FILE):
        await interaction.response.send_message("Your watchlist is empty.")
        return
    with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
        movies = [line.strip() for line in f if line.strip()]
    if not movies:
        await interaction.response.send_message("Your watchlist is empty.")
    else:
        formatted = "\n".join(f"- {m}" for m in movies)
        await interaction.response.send_message(f"**Your Watchlist:**\n{formatted}")

@bot.tree.command(name="clear_watchlist", description="Clear your entire watchlist and dataset")
async def clear_watchlist(interaction: discord.Interaction):
    open("watchlist.txt", "w").close()
    if os.path.exists("watchlist.csv"):
        os.remove("watchlist.csv")
    await interaction.response.send_message("Your watchlist has been cleared.")

@bot.tree.command(name="clear_messages", description="Clear a number of recent messages in this channel")
async def clear_messages(interaction: discord.Interaction, amount: int):
    if not interaction.user.guild_permissions.manage_messages:
        await interaction.response.send_message("You don’t have permission to manage messages.", ephemeral=True)
        return
    await interaction.response.defer()
    deleted = await interaction.channel.purge(limit=amount)
    await interaction.followup.send(f"Cleared {len(deleted)} messages.", ephemeral=True)
    
if __name__ == "__main__":
    bot.run(TOKEN)
