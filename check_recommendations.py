
import pandas as pd
from collaborative import CollaborativeRecommender

# Load the recommender
recommender = CollaborativeRecommender()

# Movies the user likes
movie_ids = [155, 27205, 1726]

# Load local MovieLens movie titles
movies = pd.read_csv("data/ml-latest-small/movies.csv")
links = pd.read_csv("data/ml-latest-small/links.csv")

links["tmdbId"] = pd.to_numeric(
    links["tmdbId"], errors="coerce"
)
links = links.dropna(subset=["tmdbId"])
links["tmdbId"] = links["tmdbId"].astype(int)

catalog = links.merge(movies, on="movieId")
title_map = dict(zip(catalog["tmdbId"], catalog["title"]))

def show_movies(label, ids):
    print(f"\n{label}")

    for movie_id in ids:
        title = title_map.get(
            movie_id, "Title not found in local catalog"
        )
        print(f"{movie_id}: {title}")

# Generate recommendations dynamically
item_results = recommender.recommend(movie_ids, n=10)
user_results = recommender.recommend_user_based(movie_ids, n=10)

show_movies("ITEM-BASED RECOMMENDATIONS", item_results)
show_movies("USER-BASED RECOMMENDATIONS", user_results)
