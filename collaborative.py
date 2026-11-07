import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE = os.path.dirname(os.path.abspath(__file__))


class CollaborativeRecommender:
    def __init__(self):
        folder = os.path.join(BASE, "model")

        # Existing item-based model.
        self.features = sp.load_npz(
            os.path.join(folder, "cf_features.npz")
        )
        self.movies = pd.read_csv(
            os.path.join(folder, "cf_movies.csv")
        )
        self.id_to_idx = {
            int(movie_id): i
            for i, movie_id in enumerate(self.movies["tmdbId"])
        }

        # New user-based model: rows are MovieLens users, columns are movies.
        self.user_ratings = sp.load_npz(
            os.path.join(folder, "user_cf_ratings.npz")
        ).tocsr()
        self.user_movies = pd.read_csv(
            os.path.join(folder, "user_cf_movies.csv")
        )
        self.user_movie_to_idx = {
            int(movie_id): i
            for i, movie_id in enumerate(self.user_movies["tmdbId"])
        }

    def recommend(self, movie_ids, n=20):
        """Existing item-based recommendations; kept compatible."""
        indexes = [
            self.id_to_idx[int(mid)]
            for mid in movie_ids
            if int(mid) in self.id_to_idx
        ]
        if not indexes:
            return []

        scores = np.zeros(self.features.shape[0], dtype=np.float32)
        for idx in indexes:
            similarities = (
                self.features @ self.features[idx].T
            ).toarray().ravel()
            scores += similarities

        scores[indexes] = -1
        top = np.argsort(-scores)[:n]
        return [
            int(self.movies.iloc[i]["tmdbId"])
            for i in top
            if scores[i] > 0
        ]

    # def recommend_user_based(self, movie_ids, n=20, min_seed_rating=3.5):
    #     """Recommend using MovieLens users who rated the selected movies well.

    #     The app supplies a list of liked movie IDs, not a real app-user ID or
    #     explicit ratings. Selected movies are therefore treated as implicit likes.
    #     """
    #     seed_indexes = [
    #         self.user_movie_to_idx[int(mid)]
    #         for mid in movie_ids
    #         if int(mid) in self.user_movie_to_idx
    #     ]
    #     if not seed_indexes:
    #         return []

    #     # For each MovieLens user, inspect ratings for the selected movies.
    #     seed_ratings = self.user_ratings[:, seed_indexes].toarray()
    #     rated_mask = seed_ratings > 0
    #     rated_counts = rated_mask.sum(axis=1)
    #     rating_sums = seed_ratings.sum(axis=1)
    #     average_seed_rating = np.divide(
    #         rating_sums,
    #         rated_counts,
    #         out=np.zeros_like(rating_sums, dtype=np.float32),
    #         where=rated_counts > 0,
    #     )

    #     # Prefer users who rated selected movies positively, with a small
    #     # coverage factor so one matching rating is not always enough.
    #     coverage = rated_counts / float(len(seed_indexes))
    #     user_weights = np.maximum(average_seed_rating - 2.5, 0) / 2.5
    #     user_weights *= coverage
    #     user_weights[average_seed_rating < min_seed_rating] = 0

    #     if not np.any(user_weights > 0):
    #         return []

    #     # Weighted average ratings from those users for every candidate movie.
    #     weighted_rating_sums = np.asarray(
    #         self.user_ratings.T @ user_weights
    #     ).ravel()
    #     rated_by_weights = np.asarray(
    #         (self.user_ratings > 0).T @ user_weights
    #     ).ravel()
    #     scores = np.divide(
    #         weighted_rating_sums,
    #         rated_by_weights,
    #         out=np.zeros_like(weighted_rating_sums, dtype=np.float32),
    #         where=rated_by_weights > 0,
    #     )

    #     # Remove the movies already in the user's list.
    #     for idx in seed_indexes:
    #         scores[idx] = -1

    #     top = np.argsort(-scores)[:n]
    #     return [
    #         int(self.user_movies.iloc[i]["tmdbId"])
    #         for i in top
    #         if scores[i] > 0
    #     ]



    
    def recommend_user_based(self, movie_ids, n=20, min_seed_rating=3.5, min_support=2):
        """Recommend movies using users who rated the selected movies positively."""

        # 1. Find selected movies in the user-based ratings matrix.
        seed_indexes = [
            self.user_movie_to_idx[int(mid)]
            for mid in movie_ids
            if int(mid) in self.user_movie_to_idx
        ]

        if not seed_indexes:
            return []

        # 2. Get ratings given by each user to the selected movies.
        seed_ratings = self.user_ratings[:, seed_indexes].toarray()
        rated_mask = seed_ratings > 0
        rated_counts = rated_mask.sum(axis=1)
        rating_sums = seed_ratings.sum(axis=1)

        average_seed_rating = np.divide(
            rating_sums,
            rated_counts,
            out=np.zeros_like(rating_sums, dtype=np.float32),
            where=rated_counts > 0,
        )

        # 3. Give higher weights to users who liked the selected movies.
        coverage = rated_counts / float(len(seed_indexes))
        user_weights = np.maximum(average_seed_rating - 2.5, 0) / 2.5
        user_weights *= coverage
        user_weights[average_seed_rating < min_seed_rating] = 0

        if not np.any(user_weights > 0):
            return []

        # 4. Calculate weighted average ratings for candidate movies.
        weighted_rating_sums = np.asarray(
            self.user_ratings.T @ user_weights
        ).ravel()

        weighted_support = np.asarray(
            (self.user_ratings > 0).T @ user_weights
        ).ravel()

        # Count how many positively weighted users rated each movie.
        active_users = (user_weights > 0).astype(np.float32)
        support_counts = np.asarray(
            (self.user_ratings > 0).T @ active_users
        ).ravel()

        scores = np.divide(
            weighted_rating_sums,
            weighted_support,
            out=np.zeros_like(weighted_rating_sums, dtype=np.float32),
            where=weighted_support > 0,
        )

        # 5. Reduce confidence in movies with little supporting evidence.
        confidence = support_counts / (support_counts + 3.0)
        scores *= confidence

        # 6. Exclude selected movies and poorly supported candidates.
        scores[support_counts < min_support] = -1

        for idx in seed_indexes:
            scores[idx] = -1

        # 7. Return the highest-scoring recommendations.
        top = np.argsort(-scores)[:n]

        return [
            int(self.user_movies.iloc[i]["tmdbId"])
            for i in top
            if scores[i] > 0
        ]
