# 🎬 Movie Recommendation System

This project implements a **Movie Recommendation System** using collaborative filtering on the MovieLens dataset. The system recommends personalized movie suggestions based on a user's previous ratings and the similarity between different films.

## 📂 GitHub Repository

[Click to view the code](https://github.com/Ahana-tabassum/CSE426_DataMining-WarehouseLAB/blob/main/MaheruTabassumOhana_2125051015_Assignment01.ipynb)

---

## 📌 Introduction

Recommendation systems play a vital role in guiding users toward content they are likely to enjoy. These systems are especially significant in digital platforms where an abundance of options can overwhelm users.

In this project, we developed a **movie recommendation system** using the MovieLens dataset, with the aim of suggesting relevant films to users based on their preferences. The core idea revolves around **collaborative filtering**, a technique that leverages user ratings to identify similarities between movies. By analyzing how users rate different films, we can detect patterns and suggest movies that align with a user's taste—even if they haven't watched those movies yet.

---

## 🛠 Methodology

### 📁 Dataset Description

Two datasets from the MovieLens collection were used:

- **ratings.csv**  
  Contains user ratings for movies.  
  Columns: `userId`, `movieId`, `rating`, `timestamp`

- **movies.csv**  
  Contains movie details.  
  Columns: `movieId`, `title`, `genres`

Both files were stored and accessed from Google Drive using Google Colab.

---

### 🧠 Movie Similarity Calculation

To recommend films similar to a user's preferences, we calculate a **similarity matrix** between movies. This matrix measures how closely related two movies are based on the ratings they receive from users. Although the actual implementation used **Pearson correlation**, the concept is aligned with **cosine similarity**.

---

### 🎯 Personalized Recommendations

The system offers personalized recommendations by identifying each user's **top-rated movies**. Then, it searches for movies that are **similar to those favorites**. Already-rated movies are filtered out, and the remaining ones are ranked by average rating from other users.

---

## 💻 Code Overview

```python
# Import necessary libraries
import pandas as pd
import numpy as np

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load datasets
ratings_data = pd.read_csv("/content/drive/MyDrive/CSE426_DataMining&WarehouseLAB/Assignment01/ratings.csv")
movies_data = pd.read_csv("/content/drive/MyDrive/CSE426_DataMining&WarehouseLAB/Assignment01/movies.csv")

# Create movie-to-movie similarity matrix
user_movie_pivot = ratings_data.pivot_table(index='userId', columns='movieId', values='rating')
similarity_matrix = user_movie_pivot.corr(method='pearson')

# Function to get similar movies
def get_similar_movies(target_movie_id, num_recommendations=5):
    if target_movie_id not in similarity_matrix:
        return "Selected movie not found in the dataset."
    similarity_scores = similarity_matrix[target_movie_id].dropna()
    top_matches = similarity_scores.sort_values(ascending=False)[1:num_recommendations+1]
    top_movies = movies_data[movies_data["movieId"].isin(top_matches.index)][["movieId", "title"]]
    return top_movies

# Example: Recommend movies similar to movieId = 1
top_recommendations = get_similar_movies(1, num_recommendations=5)

# User-based personalized recommendations
selected_user = int(input("Enter your user ID: "))
user_rated = ratings_data[ratings_data['userId'] == selected_user]

fav_movie = user_rated.loc[user_rated['rating'].idxmax()]

# Find movies not rated by the user
all_movie_ids = set(movies_data["movieId"])
rated_by_user = set(user_rated["movieId"])
not_rated_yet = all_movie_ids - rated_by_user
unseen_movies = movies_data[movies_data["movieId"].isin(not_rated_yet)]

# Recommend movies not rated by the user
def recommend_unseen_top_movies(user_id, num_movies=5):
    user_history = ratings_data[ratings_data["userId"] == user_id]
    movies_rated = set(user_history["movieId"])
    full_list = set(movies_data["movieId"])
    movies_left = full_list - movies_rated

    candidate_movies = movies_data[movies_data["movieId"].isin(movies_left)]
    avg_movie_scores = ratings_data.groupby("movieId")["rating"].mean()

    final_recommendations = candidate_movies.merge(
        avg_movie_scores, on="movieId", how="left"
    ).sort_values(by="rating", ascending=False).head(num_movies)

    return final_recommendations[["movieId", "title", "rating"]]

user_id = int(input("Enter your user ID: "))
final_suggestions = recommend_unseen_top_movies(user_id=user_id, num_movies=10)
final_suggestions

---

## ✅ Results

The system computed a **movie similarity matrix** using collaborative filtering based on user ratings, allowing it to identify films with strong correlations in viewer preferences.

### Example for User 15:
The following top 10 recommendations were generated:

| Sl. No. | Movie Title                                        | Rating |
|--------:|----------------------------------------------------|--------|
| 1       | Lamerica (1994)                                    | 5.0    |
| 2       | Won't You Be My Neighbor? (2018)                   | 5.0    |
| 3       | King of Hearts (1966)                              | 5.0    |
| 4       | Rivers and Tides (2001)                            | 5.0    |
| 5       | A Woman Under the Influence (1974)                 | 5.0    |
| 6       | What Men Talk About (2010)                         | 5.0    |
| 7       | Go for Zucker! (Alles auf Zucker!) (2004)          | 5.0    |
| 8       | 9/11 (2002)                                        | 5.0    |
| 9       | Peaceful Warrior (2006)                            | 5.0    |
| 10      | Chump at Oxford, A (1940)                          | 5.0    |

These suggestions show the system’s ability to align with a user’s interests, including non-mainstream or international films.

---

## 🧰 Tools & Technologies

- **Python 3**
- **Google Colab**
- **Pandas**, **NumPy** – for data handling
- **Pearson Correlation** – for movie similarity calculation

---

## 🧾 Conclusion

This project successfully developed a **functional and personalized movie recommendation system** using the MovieLens dataset. By applying **collaborative filtering techniques** and computing a similarity matrix between films, the system provides movie suggestions tailored to each user’s preferences.

The system enhances user experience by reducing the time spent searching for content and increasing the relevance of what is suggested.

---

## 🚀 Future Improvements

- ✅ Integrate **genre filtering** to give content-aware suggestions  
- ✅ Implement **user-user collaborative filtering**  
- ✅ Explore **Matrix Factorization (SVD)** or **Deep Learning-based** recommenders  
- ✅ Add a **frontend** or deploy as a **mini web app**
