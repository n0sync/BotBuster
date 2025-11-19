import os
import re
from dotenv import load_dotenv
import discord
from discord.ext import commands
from discord import app_commands
from src.main import run_model
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

@bot.tree.command(name="recommend", description="Get 1 recommendation based on your watchlist")
async def recommend(interaction: discord.Interaction):
    await interaction.response.defer()
    if not os.path.exists(WATCHLIST_FILE):
        await interaction.followup.send("Your watchlist is empty. Add some movies first!")
        return

    with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
        titles = [line.strip() for line in f if line.strip()]

    if not titles:
        await interaction.followup.send("Your watchlist is empty. Add some movies first!")
        return

    results = await run_model(titles, top_n=1)
    await interaction.followup.send(results[0])
    

@bot.tree.command(name="recommend_n", description="Get N recommendations based on your watchlist")
async def recommend_n(interaction: discord.Interaction, number: int):
    await interaction.response.defer()
    if not os.path.exists(WATCHLIST_FILE):
        await interaction.followup.send("Your watchlist is empty. Add some movies first!")
        return

    with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
        titles = [line.strip() for line in f if line.strip()]

    if not titles:
        await interaction.followup.send("Your watchlist is empty. Add some movies first!")
        return

    results = await run_model(titles, top_n=number)
    combined = "\n\n".join(results)

    if len(combined) <= 2000:
        await interaction.followup.send(combined)
    else:
        for i in range(0, len(combined), 1900):
            await interaction.followup.send(combined[i:i+1900])

class MovieListModal(discord.ui.Modal, title="Add Movies to Watchlist"):
    movies_input = discord.ui.TextInput(
        label="Movie Titles",
        style=discord.TextStyle.paragraph,
        placeholder="Enter movie titles (one per line or comma-separated)...",
        required=True,
        max_length=2000
    )

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()

        raw_titles = re.split(r"\d+\.\s*|,|\n|\r", self.movies_input.value)
        titles = [re.sub(r"\[.*?\]", "", t).strip(" :.-") for t in raw_titles if t.strip()]

        if not titles:
            await interaction.followup.send("No valid movie titles found in your input.")
            return

        with open(WATCHLIST_FILE, "a", encoding="utf-8") as f:
            for title in titles:
                f.write(title + "\n")

        await interaction.followup.send(f"Added {len(titles)} movies to your watchlist.")


@bot.tree.command(name="add_list", description="Paste multiple movie titles at once")
async def add_list(interaction: discord.Interaction):
    modal = MovieListModal()
    await interaction.response.send_modal(modal)


@bot.tree.command(name="add_movie", description="Add a single movie to your watchlist")
async def add_movie(interaction: discord.Interaction, movie: str):
    movie = re.sub(r"\[.*?\]", "", movie).strip(" :.-")
    if not movie:
        await interaction.response.send_message("Invalid movie title.")
        return

    with open(WATCHLIST_FILE, "a", encoding="utf-8") as f:
        f.write(movie + "\n")

    await interaction.response.send_message(f"Added **{movie}** to your watchlist.")

@bot.tree.command(name="watchlist", description="Show your current watchlist")
async def watchlist(interaction: discord.Interaction):
    if not os.path.exists(WATCHLIST_FILE):
        await interaction.response.send_message("Your watchlist is empty.")
        return
    with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
        movies = [line.strip() for line in f if line.strip()]
    if not movies:
        await interaction.response.send_message("Your watchlist is empty.")
        return
    formatted = "\n".join(f"- {m}" for m in movies)
    full_message = f"**Your Watchlist ({len(movies)} movies):**\n{formatted}"
    if len(full_message) <= 2000:
        await interaction.response.send_message(full_message)
    else:
        await interaction.response.send_message(f"**Your Watchlist ({len(movies)} movies):**")
        for i in range(0, len(formatted), 1900):
            await interaction.followup.send(formatted[i:i+1900])

@bot.tree.command(name="clear_watchlist", description="Clear your entire watchlist and dataset")
async def clear_watchlist(interaction: discord.Interaction):
    open(WATCHLIST_FILE, "w").close()
    if os.path.exists("watchlist.csv"):
        os.remove("watchlist.csv")
    await interaction.response.send_message("Your watchlist has been cleared.")

@bot.tree.command(name="clear_messages", description="Clear all messages in this channel")
async def clear_messages(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("This command can only be used in a server.", ephemeral=True)
        return
    
    if not isinstance(interaction.user, discord.Member):
        await interaction.followup.send("Unable to verify permissions.", ephemeral=True)
        return
    
    if not interaction.user.guild_permissions.manage_messages:
        await interaction.followup.send("You don't have permission to manage messages.", ephemeral=True)
        return
    
    deleted = await interaction.channel.purge(limit=None)
    await interaction.followup.send(f"Deleted {len(deleted)} messages.", ephemeral=True)
    
if __name__ == "__main__":
    bot.run(TOKEN)
