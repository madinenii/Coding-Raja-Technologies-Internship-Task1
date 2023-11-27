from flask import Flask, render_template, request
import pandas as pd
from flask import jsonify
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load movies and ratings data
movies = pd.read_csv("C:/Users/madin/Desktop/TASK1/ml-25m/movies.csv")
ratings = pd.read_csv("C:/Users/madin/Desktop/TASK1/ml-25m/ratings.csv")

# Data preprocessing for movie titles
movies["clean_title"] = movies["title"].apply(lambda x: re.sub("[^a-zA-Z0-9 ]", "", x))

# TF-IDF Vectorization for movie titles
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

# Add this route to provide movie suggestions based on user input
@app.route('/api/movies')
def movie_suggestions():
    query = request.args.get('query', '').lower()
    suggestions = [movie['title'] for movie in movies if query in movie['title'].lower()]
    return jsonify(suggestions)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    movie_title = request.form['movie_title']
    
    # Search for the movie
    title_cleaned = re.sub("[^a-zA-Z0-9 ]", "", movie_title)
    query_vec = vectorizer.transform([title_cleaned])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    
    # Get recommendations based on the selected movie
    movie_id = results.iloc[0]["movieId"]
    recommendations = find_similar_movies(movie_id)
    
    return render_template('recommendations.html', movie_title=movie_title, recommendations=recommendations)

def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

if __name__ == '__main__':
    app.run(debug=True)
