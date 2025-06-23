# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_stage_or_version, vectorizer_path):
    """
    Load model from MLflow registry and vectorizer from local path
    
    Args:
        model_name: Name of the registered model
        model_stage_or_version: Either stage name ('staging', 'production') or version number
        vectorizer_path: Local path to the vectorizer pickle file
    """
    try:
        # Set MLflow tracking URI to local mlruns directory
        mlflow_tracking_uri = os.path.join(os.getcwd(), "mlruns")
        mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")
        
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        client = MlflowClient()
        
        # Try to load model using stage/alias first, then fallback to version
        try:
            model_uri = f"models:/{model_name}/{model_stage_or_version}"
            print(f"Attempting to load model from: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Successfully loaded model from {model_uri}")
        except Exception as e:
            print(f"Failed to load model with stage/alias '{model_stage_or_version}': {e}")
            # Fallback to version number
            model_uri = f"models:/{model_name}/3"  # Use version 3 as shown in logs
            print(f"Fallback: attempting to load model from: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Successfully loaded model from {model_uri}")
        
        # Load vectorizer from local path
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        
        vectorizer = joblib.load(vectorizer_path)
        print(f"Successfully loaded vectorizer from {vectorizer_path}")
        
        return model, vectorizer
        
    except Exception as e:
        print(f"Error loading model and vectorizer: {e}")
        raise

# Initialize the model and vectorizer
try:
    # Update these paths according to your setup
    MODEL_NAME = "yt_chrome_plugin_model"  # Updated to match your registered model name
    MODEL_STAGE = "staging"  # Use staging alias as set in your logs
    VECTORIZER_PATH = "./tfidf_vectorizer.pkl"  # Update path if needed
    
    print("Loading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer(MODEL_NAME, MODEL_STAGE, VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully!")
    
except Exception as e:
    print(f"Failed to load model and vectorizer: {e}")
    model, vectorizer = None, None

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to YouTube Sentiment Analysis API",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    })

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 500
    
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments)
        
        # Handle different prediction formats
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
        
        print(f"Processed {len(comments)} comments successfully")
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [
        {
            "comment": comment, 
            "sentiment": sentiment, 
            "timestamp": timestamp
        } 
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 500
    
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments)
        
        # Handle different prediction formats
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
        
        print(f"Processed {len(comments)} comments successfully")
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [
        {
            "comment": comment, 
            "sentiment": sentiment
        } 
        for comment, sentiment in zip(comments, predictions)
    ]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(8, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'white', 'fontsize': 10}
        )
        plt.title('Sentiment Distribution', color='white', fontsize=14)
        plt.axis('equal')
        
        # Set background color
        plt.gca().set_facecolor('#1e1e1e')
        plt.gcf().patch.set_facecolor('#1e1e1e')

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', facecolor='#1e1e1e', edgecolor='none', bbox_inches='tight')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)
        
        if not text.strip():
            return jsonify({"error": "No valid text found after preprocessing"}), 400

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='#1e1e1e',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False,
            max_words=100
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['sentiment'] = df['sentiment'].astype(int)

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting with dark theme
        plt.figure(figsize=(12, 6))
        plt.style.use('dark_background')

        colors = {
            -1: '#FF6384',  # Red for negative
            0: '#C9CBCF',   # Gray for neutral
            1: '#36A2EB'    # Blue for positive
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value],
                linewidth=2,
                markersize=6
            )

        plt.title('Monthly Sentiment Percentage Over Time', color='white', fontsize=14)
        plt.xlabel('Month', color='white')
        plt.ylabel('Percentage of Comments (%)', color='white')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        
        # Set colors for ticks
        plt.tick_params(colors='white')

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', facecolor='#1e1e1e', edgecolor='none', bbox_inches='tight')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)