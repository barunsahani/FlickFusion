
import os
import pandas as pd
from collaborative import CollaborativeRecommender

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "ml-latest-small")

# Load the trained model
cf = CollaborativeRecommender()

# Load movie titles and map them to TMDB IDs
movies = pd.read_csv(os.path.join(DATA, "movies.csv"))
links = pd.read_csv(os.path.join(DATA, "links.csv"))

links = links.dropna(subset=["tmdbId"]).copy()
links["tmdbId"] = links["tmdbId"].astype(int)

catalog = movies.merge(links, on="movieId")
title_map = dict(zip(catalog["tmdbId"], catalog["title"]))

# Select a movie that exists in the trained model
test_movie = int(cf.movies.iloc[0]["tmdbId"])
movie_title = title_map.get(test_movie, "Unknown movie")

print("Testing movie:", movie_title)
print("\nRecommended movies:")

recommendations = cf.recommend([test_movie], n=10)

if not recommendations:
    print("No recommendations found.")
else:
    for i, movie_id in enumerate(recommendations, 1):
        print(f"{i}. {title_map.get(movie_id, movie_id)}")

    print("\nSUCCESS: Recommendations generated!")
