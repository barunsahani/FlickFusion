
from collaborative import CollaborativeRecommender

# Load both recommendation models
recommender = CollaborativeRecommender()

# Three movies that the user likes
movie_ids = [
    155,    # The Dark Knight
    27205,  # Inception
    1726    # Iron Man
]

print("Selected movies:", movie_ids)

print("\nItem-based recommendations:")
item_movies = recommender.recommend(movie_ids, n=10)
print(item_movies)

print("\nUser-based recommendations:")
user_movies = recommender.recommend_user_based(movie_ids, n=10)
print(user_movies)
