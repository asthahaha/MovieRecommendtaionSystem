import flask
import openpyxl
from flask import request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = flask.Flask(__name__)

# Load the movie dataset from an Excel file
try:
    movies_df = pd.read_excel('C:\\Users\\astha\\IdeaProjects\\astha\\movies.xlsx', engine='openpyxl')
except pd.errors.ParserError:
    movies_df = pd.read_excel('C:\\Users\\astha\\IdeaProjects\\astha\\movies.xlsx', engine='openpyxl', error_bad_lines=False)

# Preprocess the data
tfidf = TfidfVectorizer(stop_words='english')
movies_df['overview'] = movies_df['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_df['overview'])

# Compute similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = movies_df[movies_df['title'] == title].index[0]
    except IndexError:
        return []  # Return an empty list if the title is not found
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()

# API endpoint for movie recommendation
@app.route('/recommend', methods=['GET'])
def recommend_movies():
    title = request.args.get('title')
    if title:
        recommended_movies = get_recommendations(title)
        return jsonify(movies=recommended_movies)
    else:
        return jsonify(error='Please provide a movie title')

if __name__ == '__main__':
    app.run(debug=True)
