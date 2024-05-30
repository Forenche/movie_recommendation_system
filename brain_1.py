# Import required modules
import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, jsonify, request, render_template
import requests

file_path = "/home/forenche/Music/movie_dataset.csv"

data = []
with open(file_path, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    try:
        for row in csv_reader:
            data.append(row)
    except csv.Error as e:
        pass

# Setting up our dataframe
df = pd.DataFrame(data)

print("Printing columns",df.columns)

# Printing for sanity

print(df)

print("DESCRIE\n")
print(df.describe)
print("HEAD\n")
print(df.head)
print("SHAPE\n")
print(df.shape)

print("DESCRBE AFTER DOING NULL\n")
df.isnull().sum()
print(df.describe)

print("Describe after null null\n")
df = df.dropna().copy()
print("HEAD\n")
print(df.head)

df.columns = ['movieId', 'title', 'genres', 'rating']
genress = df['genres'].str.get_dummies(sep='|')
genress.columns = [f'genres_{genres}' for genres in genress.columns]

df = pd.concat([df, genress], axis=1)

# Pop 'genres_genres' as it is a placeholder
df.pop('genres_genres')

df.drop(columns=['genres'], inplace=True)
print("HEAD after dropping genres\n")
print(df.head)

print("printing columns\n")
print(df.columns)

print("DESCRIBE\n")
print(df.describe)

print("SHAPE\n")
print(df.shape)

# Print missing values for sanity
missing_values = df.isnull().sum()
print("missing_values:\n",missing_values)

df['title'].fillna('Unknown', inplace=True)
print("data type: \n",df.dtypes)

print("printing columns\n")
print(df.columns)
df['movieId'] = pd.to_numeric(df['movieId'], errors='coerce')
print(df.head)

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
print("Printing rating: \n",df.rating)

print(df.columns)

# Fetch genre columns
genre_columns = [col for col in df.columns if col.startswith('genres_')]

movie_ratings_pivot = df.pivot_table(index='movieId', values=genre_columns, aggfunc='mean')

movie_ratings_pivot.fillna(0, inplace=True)

# Calculate cosine similarity between movies based on their genre ratings
movie_similarity = cosine_similarity(movie_ratings_pivot)

# Genre ID to name mapping (Ref: https://developer.themoviedb.org/reference/genre-movie-list)
genre_id_to_name = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western"
}

# Function to preprocess the user input and extract genre IDs
def extract_genre_ids_from_text(text):
    name_to_genre_id = {genre_name.lower(): genre_id for genre_id, genre_name in genre_id_to_name.items()}
    extracted_genre_ids = []
    tokens = text.lower().split()
    for token in tokens:
        genre_id = name_to_genre_id.get(token)
        if genre_id:
            extracted_genre_ids.append(genre_id)
    
    return extracted_genre_ids

# Function to recommend movies based on user preferences and genre similarity
def recommend_movies(genres_to_recommend):
    recommended_movies = []
    for genre in genres_to_recommend:
        # Find movies with the highest similarity to the selected genre
        similar_movies_indices = movie_similarity[:, genre].argsort()[::-1]
        # Exclude the selected genre itself
        similar_movies_indices = similar_movies_indices[similar_movies_indices != genre]
        # Get recommended movies for this genre
        recommended_movies.extend([genre_id_to_name.get(idx) for idx in similar_movies_indices])
    return recommended_movies[10]

def fetch_latest_movies(genre_ids):
    try:
        # Read API key from text file, no leaking keys
        with open("keys.txt") as keys_file:
            API_KEY = keys_file.readline().rstrip()

        GENRES_ID_STR = "%2C".join(str(genre_id) for genre_id in genre_ids)
        url = f"https://api.themoviedb.org/3/discover/movie?include_adult=true&include_video=false&language=en-US&page=1&sort_by=popularity.desc&with_genres={GENRES_ID_STR}"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        # Fetch data from the API
        response = requests.get(url, headers=headers)
        print("Response status code:", response.status_code)  # Debugging statement
        data = response.json()
        print("Data:", data)  # Debugging statement

        # Extract results from the response
        latest_movies = data.get('results', [])
        return latest_movies

    except requests.exceptions.RequestException as e:
        print(f"Error fetching latest movies: {e}")
        return []

def fetch_movie_ratings(genre_ids):
    try:
        # Read API key from text file, no leaking keys
        with open("keys.txt") as keys_file:
            API_KEY = keys_file.readline().rstrip()

        # Construct the URL with the provided genre IDs
        GENRES_ID_STR = "%2C".join(str(genre_id) for genre_id in genre_ids)
        url = f"https://api.themoviedb.org/3/discover/movie?include_adult=true&include_video=false&language=en-US&page=1&sort_by=popularity.desc&with_genres={GENRES_ID_STR}"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        # Fetch data from the API
        response = requests.get(url, headers=headers)
        print("Response status code:", response.status_code)  # Debugging statement
        data = response.json()
        print("Data:", data)  # Debugging statement

        # Extract movie IDs and ratings from the response
        movie_ratings = {}
        for movie in data.get('results', []):
            movie_id = movie.get('id')
            rating = movie.get('vote_average')
            if movie_id and rating:
                adj_rating = 0.5 * rating
                movie_ratings[str(movie_id)] = adj_rating
        
        # Create a pandas Series with movie IDs as index and ratings as values
        movie_ratings_series = pd.Series(movie_ratings, name='rating')
        return movie_ratings_series
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movie ratings: {e}")
        return pd.Series()  # Return an empty Series in case of an error

def preprocess_latest_movies(latest_movies, movie_ratings):
    movie_descriptions = []
    ratings = []

    for movie in latest_movies:
        # Extract relevant features (title, overview, genres) and concatenate them
        description = f"{movie['title']} {movie.get('overview', '')} {' '.join([genre_id_to_name.get(genre_id, '') for genre_id in movie.get('genre_ids', [])])}"
        # Check if the description is empty or contains only stop words
        if description.strip():
            movie_descriptions.append(description)
            # Get the rating for the current movie ID
            ratings.append(movie_ratings.get(str(movie['id']), 0.0))

    if not movie_descriptions:
        return None

    vectorizer = CountVectorizer(stop_words='english')
    movie_vectors = vectorizer.fit_transform(movie_descriptions)
    tokenized_words = vectorizer.get_feature_names_out()

    # Calculate average rating for each genre
    genre_ratings = {}
    for movie in latest_movies:
        for genre_id in movie.get('genre_ids', []):
            genre_name = genre_id_to_name.get(genre_id)
            if genre_name:
                genre_ratings.setdefault(genre_name, []).append(movie_ratings.get(str(movie['id']), 0.0))

    avg_genre_ratings = {genre: sum(ratings) / len(ratings) for genre, ratings in genre_ratings.items()}
    avg_ratings = [avg_genre_ratings.get(genre_id_to_name.get(genre_id), 0.0) for movie in latest_movies for genre_id in movie.get('genre_ids', [])]
    combined_vectors = pd.concat([pd.DataFrame(movie_vectors.toarray(), columns=tokenized_words), pd.Series(ratings, name='rating'), pd.Series(avg_ratings, name='avg_rating')], axis=1)
    combined_vectors = combined_vectors.fillna(0)

    return combined_vectors

# Function to recommend movies based on a list of genres
def recommend_movies_by_genres(genres, latest_movies, movie_ratings):
    latest_movie_titles = [movie['title'] for movie in latest_movies]
    combined_vectors = preprocess_latest_movies(latest_movies, movie_ratings)

    if combined_vectors is None:
        return [], latest_movie_titles

    recommended_movies = []
    
    # Calculate cosine similarity for each genre separately
    for genre in genres:
        genre_name = genre_id_to_name.get(genre, "")
        if not genre_name:
            continue
        
        genre_movies = [movie for movie in latest_movies if genre_name in [genre_id_to_name.get(g, "") for g in movie.get('genre_ids', [])]]
        genre_indices = [i for i, movie in enumerate(latest_movies) if genre_name in [genre_id_to_name.get(g, "") for g in movie.get('genre_ids', [])]]
        print(f"Genre: {genre_name}, Number of Movies: {len(genre_movies)}, Genre Indices: {genre_indices}")
        print(f"Number of movies being considered for {genre_name}: {len(genre_movies)}")
        genre_vectors = combined_vectors.iloc[genre_indices]
        print(f"Genre Vectors:\n{genre_vectors}")    
        if not genre_movies:
            continue

        genre_similarity = cosine_similarity(genre_vectors)
        print("Genre similarity matrix:", genre_similarity)  # Debugging
        
        # Select top recommendations for this genre
        for i in range(len(genre_movies)):
            similar_movies_indices = genre_similarity[i].argsort()[::-1]
            similar_movies_indices = [index for index in similar_movies_indices if index != i]
            similar_movies_indices = [index for index in similar_movies_indices if 0 <= index < len(genre_movies)]
            recommended_movies.extend([genre_movies[index]['title'] for index in similar_movies_indices])

    return recommended_movies, latest_movie_titles

app = Flask(__name__, template_folder="templates")

@app.route('/recommend_movies', methods=['GET'])
def get_recommendations():
    text_prompt = request.args.get('text_prompt')
    if not text_prompt:
        return jsonify({'error': 'Text prompt is required'}), 400
    
    # Extract genres from the text prompt
    genres_to_recommend = extract_genre_ids_from_text(text_prompt)
    latest_movies = fetch_latest_movies(genres_to_recommend)
    print("Latest movies:", latest_movies)  # Debugging statement
    movie_ratings = fetch_movie_ratings(genres_to_recommend)
    print("Movies ratings:", movie_ratings)  # Debugging statement

    recommended_movies, latest_movie_titles = recommend_movies_by_genres(genres_to_recommend, latest_movies, movie_ratings)

    print("Recommended movies:", recommended_movies)  # Debugging statement

    response_data = {
        'recommended_movies': recommended_movies,
        'latest_movie_titles': latest_movie_titles
    }

    return jsonify(response_data)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
