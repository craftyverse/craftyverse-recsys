import numpy as np
import pandas as pd

def CF_cosine_recommender(user, user_similarity_matrix, ratings_matrix, n_recommendations=5):
    """
    User-Based Collaborative Filtering Recommender using Cosine Similarity.

    Parameters:
    - user: The user ID for whom to generate recommendations.
    - user_similarity_matrix: A DataFrame containing user-user cosine similarity scores.
    - ratings_matrix: A DataFrame containing user-item ratings.
    - n_recommendations: Number of recommendations to generate.

    Returns:
    - A list of recommended item IDs.
    """

    # Get similarity scores for the target user
    similarity_scores = user_similarity_matrix.loc[user]

    # Get the weighted sum of ratings for similar users
    weighted_ratings = np.dot(similarity_scores, ratings_matrix)

    # Get the sum of similarity scores
    sum_of_similarities = np.sum(similarity_scores)

    # Calculate the weighted average ratings
    weighted_avg_ratings = weighted_ratings / sum_of_similarities

    # Create a Series for the weighted average ratings
    weighted_avg_ratings_series = pd.Series(weighted_avg_ratings, index=ratings_matrix.columns)

    # Sort the series to get top N recommendations
    sorted_recommendations = weighted_avg_ratings_series.sort_values(ascending=False)

    # Get the top N recommendations
    recommendations = sorted_recommendations.index[:n_recommendations].tolist()

    return recommendations
