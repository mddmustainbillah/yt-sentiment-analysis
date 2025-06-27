import pytest
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import pickle
import os
import numpy as np

@pytest.mark.parametrize("model_name, stage, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "tfidf_vectorizer.pkl"),
])
def test_model_with_vectorizer(model_name, stage, vectorizer_path):
    client = MlflowClient()

    try:
        # Get the model using alias or version (matching your register_model.py logic)
        try:
            model_version = client.get_model_version_by_alias(model_name, stage)
            model_uri = f"models:/{model_name}@{stage}"
            version_info = model_version.version
            print(f"Using alias-based model URI: {model_uri} (version {version_info})")
        except Exception as alias_error:
            print(f"Could not use alias approach: {alias_error}")
            # Fallback to stage-based approach
            latest_version_info = client.get_latest_versions(model_name, stages=["Staging"])
            assert latest_version_info, f"No model found in staging for '{model_name}'"
            version_info = latest_version_info[0].version
            model_uri = f"models:/{model_name}/{version_info}"
            print(f"Using version-based model URI: {model_uri}")

        # Load the model
        model = mlflow.pyfunc.load_model(model_uri)

        # Load the vectorizer (check if file exists)
        if not os.path.exists(vectorizer_path):
            pytest.fail(f"Vectorizer file not found: {vectorizer_path}")

        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        # Create test input (matching your model_evaluation.py approach)
        test_comments = [
            "This video is amazing!",
            "Not very good content",
            "Average video, could be better"
        ]
        print(f"Testing with {len(test_comments)} sample comments")

        # Transform the input using vectorizer (matching model_evaluation.py)
        input_data = vectorizer.transform(test_comments)
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())

        print(f"Input shape after vectorization: {input_df.shape}")

        # Predict using the model
        predictions = model.predict(input_df)

        # Verify the input and output shapes
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"
        assert len(predictions) == len(test_comments), "Output count mismatch"

        # Verify predictions are valid (assuming 3 classes: -1, 0, 1 based on your model)
        unique_predictions = set(predictions)
        print(f"Unique predictions: {unique_predictions}")

        # Check that predictions are valid numeric types (including numpy types)
        valid_predictions = all(isinstance(pred, (int, float, np.integer, np.floating)) for pred in predictions)
        assert valid_predictions, f"Invalid prediction types found: {[type(pred) for pred in predictions]}"

        # Additional validation: Check if predictions are in expected sentiment range
        # Assuming sentiment classes are typically -1 (negative), 0 (neutral), 1 (positive)
        # or 0 (negative), 1 (neutral), 2 (positive) - adjust based on your model
        prediction_values = [int(pred) for pred in predictions]  # Convert to regular int for easier checking
        
        # You can customize this range based on your actual model's output classes
        expected_range = range(-1, 2)  # -1, 0, 1 for sentiment
        # Alternative: expected_range = range(0, 3)  # 0, 1, 2 for sentiment
        
        valid_range = all(pred in expected_range for pred in prediction_values)
        if not valid_range:
            print(f"Warning: Some predictions outside expected range {list(expected_range)}: {prediction_values}")
            # Don't fail the test for this, just warn, as the model might use different encoding

        print(f"✅ Model '{model_name}' version {version_info} signature test passed.")
        print(f"Sample predictions: {predictions}")
        print(f"Prediction types: {[type(pred) for pred in predictions]}")

    except Exception as e:
        pytest.fail(f"Model signature test failed with error: {e}")


if __name__ == "__main__":
    # For running the test directly
    test_model_with_vectorizer("yt_chrome_plugin_model", "staging", "tfidf_vectorizer.pkl")