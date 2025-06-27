# scripts/test_model_performance.py
import pytest
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.tracking import MlflowClient

# Set local MLflow tracking URI (matching your setup)
mlflow.set_tracking_uri("/Users/mustainbillah/Desktop/yt-sentiment-analysis/mlruns")

@pytest.mark.parametrize("model_name, stage, holdout_data_path, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl"),
])
def test_model_performance(model_name, stage, holdout_data_path, vectorizer_path):
    client = MlflowClient()
    
    try:
        # Get the model using alias or version (matching your register_model.py approach)
        try:
            model_version = client.get_model_version_by_alias(model_name, stage)
            model_uri = f"models:/{model_name}@{stage}"
            version_info = model_version.version
            print(f"Using alias-based model URI: {model_uri}")
        except:
            # Fallback to stage-based approach
            latest_version_info = client.get_latest_versions(model_name, stages=["Staging"])
            assert latest_version_info, f"No model found in staging for '{model_name}'"
            version_info = latest_version_info[0].version
            model_uri = f"models:/{model_name}/{version_info}"
            print(f"Using version-based model URI: {model_uri}")
        
        # Load the model (using pyfunc like your other tests)
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Load the vectorizer
        if not os.path.exists(vectorizer_path):
            pytest.fail(f"Vectorizer file not found: {vectorizer_path}")
            
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        
        # Load the holdout test data
        if not os.path.exists(holdout_data_path):
            pytest.fail(f"Test data file not found: {holdout_data_path}")
            
        holdout_data = pd.read_csv(holdout_data_path)
        print(f"Test data loaded with shape: {holdout_data.shape}")
        
        # Handle missing values (matching your model_evaluation.py approach)
        holdout_data.fillna('', inplace=True)
        
        # Prepare test data (matching your model_evaluation.py structure)
        X_holdout_raw = holdout_data['clean_comment'].values
        y_holdout = holdout_data['category'].values
        
        # Apply TF-IDF transformation (matching your model_evaluation.py approach)
        X_holdout_tfidf = vectorizer.transform(X_holdout_raw)
        X_holdout_tfidf_df = pd.DataFrame(X_holdout_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        
        # Predict using the model
        y_pred = model.predict(X_holdout_tfidf_df)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_holdout, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_holdout, y_pred, average='weighted', zero_division=1)
        
        # Define performance thresholds (adjust based on your requirements)
        expected_accuracy = 0.70    # 70% accuracy threshold
        expected_precision = 0.65   # 65% precision threshold
        expected_recall = 0.65      # 65% recall threshold
        expected_f1 = 0.65          # 65% F1 score threshold
        
        # Print performance metrics
        print(f"\nModel Performance Metrics for version {version_info}:")
        print(f"Accuracy: {accuracy:.4f} (threshold: {expected_accuracy})")
        print(f"Precision: {precision:.4f} (threshold: {expected_precision})")
        print(f"Recall: {recall:.4f} (threshold: {expected_recall})")
        print(f"F1 Score: {f1:.4f} (threshold: {expected_f1})")
        
        # Assert that the model meets the performance thresholds
        assert accuracy >= expected_accuracy, f'Accuracy {accuracy:.4f} below threshold {expected_accuracy}'
        assert precision >= expected_precision, f'Precision {precision:.4f} below threshold {expected_precision}'
        assert recall >= expected_recall, f'Recall {recall:.4f} below threshold {expected_recall}'
        assert f1 >= expected_f1, f'F1 score {f1:.4f} below threshold {expected_f1}'
        
        print(f"✅ Performance test passed for model '{model_name}' version {version_info}")
        
    except Exception as e:
        pytest.fail(f"Model performance test failed with error: {e}")

if __name__ == "__main__":
    test_model_performance("yt_chrome_plugin_model", "staging", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl")