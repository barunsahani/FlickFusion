import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "ml-latest-small")
OUT = os.path.join(BASE, "model")

ratings = pd.read_csv(os.path.join(DATA, "ratings.csv"))
links = pd.read_csv(os.path.join(DATA, "links.csv"))

# Connect MovieLens movie IDs to TMDB IDs.
links = links.dropna(subset=["tmdbId"]).copy()
links["tmdbId"] = links["tmdbId"].astype(int)
ratings = ratings.merge(links[["movieId", "tmdbId"]], on="movieId")
ratings = ratings.drop_duplicates(subset=["userId", "tmdbId"])

movie_ids = np.sort(ratings["tmdbId"].unique())
user_ids = np.sort(ratings["userId"].unique())
movie_index = {int(mid): i for i, mid in enumerate(movie_ids)}
user_index = {int(uid): i for i, uid in enumerate(user_ids)}

rows = ratings["tmdbId"].map(movie_index).to_numpy()
cols = ratings["userId"].map(user_index).to_numpy()
values = ratings["rating"].to_numpy(dtype=np.float32)

# Movie x User rating matrix: used by the existing item-based model.
R = sp.csr_matrix(
    (values, (rows, cols)),
    shape=(len(movie_ids), len(user_ids)),
    dtype=np.float32,
)

# Keep existing item-based model outputs compatible.
item_features = normalize(R, norm="l2", axis=1)
os.makedirs(OUT, exist_ok=True)
sp.save_npz(os.path.join(OUT, "cf_features.npz"), item_features)
pd.DataFrame({"tmdbId": movie_ids}).to_csv(
    os.path.join(OUT, "cf_movies.csv"), index=False
)

# User x Movie raw rating matrix: used by user-based recommendations.
user_ratings = R.T.tocsr()
sp.save_npz(os.path.join(OUT, "user_cf_ratings.npz"), user_ratings)
pd.DataFrame({"userId": user_ids}).to_csv(
    os.path.join(OUT, "user_cf_users.csv"), index=False
)
pd.DataFrame({"tmdbId": movie_ids}).to_csv(
    os.path.join(OUT, "user_cf_movies.csv"), index=False
)

print("Item-based and user-based matrices saved!")
print("Movies:", len(movie_ids))
print("Users:", len(user_ids))
