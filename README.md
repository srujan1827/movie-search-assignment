# movie-search-assignment
Semantic search on movie plots using SentenceTransformers (Assignment 1).


Repository contains my solution for "Assignment-1: Semantic Search on Movie Plots"

Overview
Built a semantic search engine for movie plots using **SentenceTransformers (all-MiniLM-L6-v2)**.  
The system encodes movie plots into embeddings and retrieves the most relevant ones for a given query using **cosine similarity**.

Setup
1. Clone this repo:
   ```bash
   git clone https://github.com/srujan1827/movie-search-assignment.git
   cd movie-search-assignment
2. Install requirements
   
       pip install -r requirements.txt
3.Run the notebook

       jupyter notebook AI_Systems_Development_Assignment_1_srujan_avasarala_.ipynb
4. Testing
   
       python -m unittest test_movie_search.py -v
6. Usage Example
   
       from movie_search import search_movies
       print(search_movies("spy thriller in Paris", top_n=3))



