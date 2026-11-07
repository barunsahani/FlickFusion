
# 🎬 FLICKFUSION

### Discover Movies. Match Your Taste. Experience More.

**A Hybrid Movie Recommendation System Using Collaborative and
Content-Based Filtering**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikitlearn)](https://scikit-learn.org/)


------------------------------------------------------------------------

## 📌 About the Project

**FLICKFUSION** is a Netflix-inspired movie discovery and recommendation
web application that helps users find movies based on their interests
and selected favourites.

The application combines a responsive, single-page frontend with a
Python/FastAPI backend and a hybrid recommendation engine. Instead of
relying on only one recommendation technique, FLICKFUSION combines:

-   **Content-Based Filtering** -- finds movies with similar content and
    attributes.
-   **Item-Based Collaborative Filtering** -- recommends movies related
    to items in the user's selected movie list.
-   **User-Based Collaborative Filtering** -- uses patterns from users
    with similar movie preferences.

The three ranked recommendation lists are combined using **weighted
Reciprocal Rank Fusion (RRF)** to produce a final personalized list.

## ✨ Features

-   Netflix-inspired interface with a featured hero banner.
-   Horizontally scrolling movie collections and genre rows.
-   Movie search with local catalog results and optional live TMDB
    search.
-   Movie detail view with related recommendations.
-   **More Like This** recommendations.
-   **My List** for saving selected movies and generating personalized
    picks.
-   Hybrid recommendations using content-based, item-based, and
    user-based models.
-   Movie posters and backdrops refreshed through TMDB when an API key
    is configured.
-   Local movie catalog and pre-trained model artifacts.
-   FastAPI endpoints for the frontend and recommendation workflows.

## 🧠 Recommendation System

FLICKFUSION combines three recommendation approaches:

  -----------------------------------------------------------------------
  Technique                          |   How it works
  ----------------------------------- -----------------------------------
  Content-Based Filtering  :          Builds a movie profile from
                                      attributes such as overview,
                                      genres, production companies,
                                      language, release era, title words,
                                      and franchise information.
                                      Similarity is calculated using
                                      feature representations.

  Item-Based Collaborative Filtering : Uses item-feature relationships to
                                      find movies related to the movies
                                      selected by the user.

  User-Based Collaborative Filtering : Uses user preferences patterns to 
                                      identify movies associated with similar user tastes.

  -----------------------------------------------------------------------

### Hybrid ranking

The `/api/hybrid-profile` endpoint generates recommendations from all
three models and merges their ranked results using weighted Reciprocal
Rank Fusion.

Current weights in `main.py`:

  Recommendation source        Weight
  -------------------------- --------
  Content-based                  0.45
  Item-based collaborative       0.30
  User-based collaborative       0.25

The system excludes movies already selected by the user and filters
recommendations to movies available in the local catalog before
returning the final ranked results.

> These are the current weights defined in the source code. They can be
> adjusted in `main.py` as the recommendation system is evaluated and
> improved.

## 🛠️ Tech Stack

### Backend and Machine Learning

-   Python
-   FastAPI
-   Uvicorn
-   Pandas and NumPy
-   SciPy
-   scikit-learn
-   Pydantic
-   HTTPX

### Frontend

-   HTML
-   CSS
-   Vanilla JavaScript

### Data and External Services

-   MovieLens dataset
-   Movie metadata used to build the local catalog
-   The Movie Database (TMDB) API -- optional, for current movie
    metadata, posters, backdrops, and live movie discovery

## 📁 Project Structure

``` text
FlickFusion/
│
├── data/
│   └── ml-latest-small/       # MovieLens dataset files
│
├── model/                     # Pre-trained model and feature artifacts
│
├── static/
│   ├── index.html             # Main frontend page
│   ├── style.css              # Application styling
│   └── app.js                 # Frontend interactions and API calls
│
├── main.py                    # FastAPI application and API endpoints
├── recommender.py             # Content-based recommendation logic
├── collaborative.py           # Collaborative filtering models
├── train.py                   # Content model training and evaluation
├── train_cf.py                # Collaborative model training
├── check_recommendations.py   # Recommendation inspection utility
├── test_cf.py                 # Item-based CF checks
├── test_user_cf.py            # User-based CF checks
├── poster_cache.json          # Cached poster/backdrop information
├── requirements.txt           # Python dependencies
├── runtime.txt                # Runtime configuration
├── Procfile                   # Process configuration for deployment
└── README.md
```

## ⚙️ Getting Started

Follow these steps to run FLICKFUSION locally.

### 1. Clone the repository

``` bash
git clone https://github.com/barunsahani/FlickFusion.git
cd FlickFusion
```

### 2. Create a virtual environment

**Windows (PowerShell):**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure TMDB (optional)

FLICKFUSION can use its local catalog and pre-trained model without a
TMDB API key. To retrieve refreshed posters, backdrops, and live movie
information, configure a TMDB API key.

1.  Create an account at [The Movie Database
    (TMDB)](https://www.themoviedb.org/).
2.  Request an API key from your account's API settings.
3.  Create a `.env` file in the project root.
4.  Add:

``` env
TMDB_API_KEY=your_tmdb_api_key
```

5.  Restart the application after saving the file.

**Keep your API key private.** Do not commit `.env` or expose your key
in frontend code. If no key is configured, TMDB-dependent live features
and refreshed poster lookups will be unavailable; local catalog
functionality remains available.

### 5. Run the application

``` bash
uvicorn main:app --reload
```

Open the application in your browser:

``` text
http://127.0.0.1:8000
```

FastAPI's interactive API documentation is available at:

``` text
http://127.0.0.1:8000/docs
```

## 🔌 API Endpoints

The backend exposes endpoints used by the frontend and recommendation
workflows.

  --------------------------------------------------------------------------
  Method                  Endpoint                   Purpose
  ----------------------- -------------------------- -----------------------
  GET                     `/`                        Serves the frontend
                                                     application

  GET                     `/api/health`              Checks backend status
                                                     and catalog information

  GET                     `/api/home`                Returns the hero movie,
                                                     home rows, and genres

  GET                     `/api/genre/{name}`        Returns movies from a
                                                     selected genre

  GET                     `/api/search?q={query}`    Searches the movie
                                                     catalog

  GET                     `/api/movie/{movie_id}`    Returns movie details
                                                     and similar
                                                     recommendations

  POST                    `/api/profile`             Returns content-based
                                                     picks for selected
                                                     movie IDs

  POST                    `/api/hybrid-profile`      Returns hybrid
                                                     personalized
                                                     recommendations

  POST                    `/api/cards`               Returns movie cards for
                                                     supplied movie IDs

  GET                     `/api/live`                Returns live TMDB movie
                                                     rows when configured

  GET                     `/api/indian-movies`       Returns popular Indian
                                                     movies from TMDB

  GET                     `/api/poster/{movie_id}`   Retrieves refreshed
                                                     poster information

  POST                    `/api/posters`             Retrieves poster
                                                     information for
                                                     multiple movie IDs
  --------------------------------------------------------------------------

### Example: Hybrid recommendation request

``` http
POST /api/hybrid-profile
Content-Type: application/json
```

Request body:

``` json
{
  "ids": [155, 27205, 1726],
  "n": 20
}
```

-   `ids`: TMDB movie IDs selected by the user.
-   `n`: Maximum number of recommendations requested.

The endpoint returns a list of movie cards and identifies the
recommendation method as `hybrid`.

## 🔁 Retraining the Models (Optional)

The repository includes scripts for training the recommendation models.
The pre-trained artifacts are already stored in the `model/` directory,
so retraining is not required for the normal local startup workflow.

### Content-based model

The content-model training script can use `movies_metadata.csv` from
Kaggle's *The Movies Dataset*.

Place the CSV next to `train.py`, then run:

``` bash
python train.py --csv movies_metadata.csv
```

The script builds and evaluates the content-based model and writes
updated artifacts to the `model/` directory.

### Collaborative filtering models

The repository also includes `train_cf.py` for collaborative filtering
model preparation. Check the script's expected input files and paths
before running it:

``` bash
python train_cf.py
```

## 🧪 Testing and Inspection

The repository contains scripts to inspect recommendation output and
test collaborative filtering:

``` bash
python check_recommendations.py
python test_cf.py
python test_user_cf.py
```

Run these from the project root after installing dependencies and
ensuring the required model artifacts and dataset files are present.

## 📊 Model Evaluation

The content-based model includes an evaluation report at
`model/report.json`. The existing project evaluation uses proxy metrics
such as:

-   Franchise Recall@10
-   Franchise Hit@10
-   Genre Jaccard@10

**Important:** These metrics evaluate similarity against a limited proxy
(franchise relationships); they should not be interpreted as proof of
real-world user satisfaction or recommendation accuracy. A stronger
evaluation would use held-out user ratings, user feedback, and ranking
metrics such as Precision@K, Recall@K, MAP, or NDCG.

## 🚀 Deployment

A `Procfile` is included for process-based hosting platforms. A typical
web process command is:

``` text
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Before deploying, verify the platform's Python runtime,
environment-variable configuration, persistent storage behavior, and
availability of all files in the `model/` and `data/` directories.
Configure `TMDB_API_KEY` as a secret environment variable if live TMDB
features are required.

## 🔮 Future Improvements

-   Collect explicit ratings, likes, and watch history from users.
-   Improve cold-start recommendations for new users.
-   Evaluate the hybrid model with user-based train/test splits and
    ranking metrics.
-   Add authentication and cloud-based user profiles.
-   Persist **My List** across devices and sessions.
-   Add filters for language, release year, rating, and genre.
-   Improve caching and response times for live movie data.
-   Add automated tests for API endpoints and recommendation quality.

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome.

1.  Fork the repository.
2.  Create a feature branch.
3.  Make your changes and test them locally.
4.  Commit your changes with a clear message.
5.  Open a pull request describing your updates.

## 👨‍💻 Author

**Barun Sahani** | **Deepanshu Harsh** | **Ayush Shrivastav** | **Abhishek Chaudhary**

-   GitHub: [@barunsahani](https://github.com/barunsahani)
-   Project: [FLICKFUSION](https://github.com/barunsahani/FlickFusion)

## 🙏 Acknowledgements

-   [MovieLens](https://grouplens.org/datasets/movielens/) for movie
    rating data.
-   [The Movie Database (TMDB)](https://www.themoviedb.org/) for movie
    metadata and images.
-   [FastAPI](https://fastapi.tiangolo.com/) and the Python open-source
    ecosystem.

------------------------------------------------------------------------

**FLICKFUSION --- Your taste. Your movies. Your next favourite.**

