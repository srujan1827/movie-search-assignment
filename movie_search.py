import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
movies_df = pd.read_csv("movies.csv")
movies_df["embedding"] = movies_df["plot"].apply(lambda x: model.encode(x))

def search_movies(query: str, top_n: int = 5) -> pd.DataFrame:
    """
    Search for movies most similar to the query based on cosine similarity.

    Args:
        query (str): The search string
        top_n (int): Number of top results to return

    Returns:
        pd.DataFrame: DataFrame with columns [title, plot, similarity]
    """
    query_embedding = model.encode([query])
    similarities = cosine_similarity(list(movies_df["embedding"]), query_embedding)
    movies_df["similarity"] = similarities
    results = movies_df.sort_values("similarity", ascending=False).head(top_n)
    return results[["title", "plot", "similarity"]]
