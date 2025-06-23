# src/model/model_evaluation.py
import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature
import tempfile
import shutil

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()
    
    # Clean up the local file
    if os.path.exists(cm_file_path):
        os.remove(cm_file_path)


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    # Set MLflow tracking URI
    tracking_uri = "/Users/mustainbillah/Desktop/yt-sentiment-analysis/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('dvc-pipeline-runs')
    
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    
    with mlflow.start_run() as run:
        try:
            logger.info(f"Started MLflow run: {run.info.run_id}")
            
            # Load parameters from YAML file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Load model and vectorizer
            model_pkl_path = os.path.join(root_dir, 'lgbm_model.pkl')
            vectorizer_pkl_path = os.path.join(root_dir, 'tfidf_vectorizer.pkl')
            
            logger.info(f"Loading model from: {model_pkl_path}")
            logger.info(f"Loading vectorizer from: {vectorizer_pkl_path}")
            
            model = load_model(model_pkl_path)
            vectorizer = load_vectorizer(vectorizer_pkl_path)

            # Load test data
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            logger.info(f"Test data shape: {test_data.shape}")

            # Prepare test data
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Create input example for signature
            # Use a small sample of the actual TF-IDF transformed data
            sample_size = min(5, X_test_tfidf.shape[0])
            X_sample = X_test_tfidf[:sample_size]
            y_sample = model.predict(X_sample)
            
            # Convert sparse matrix to dense for signature
            X_sample_dense = X_sample.toarray()
            
            # Infer signature
            signature = infer_signature(X_sample_dense, y_sample)
            logger.info("Model signature inferred successfully")

            # CRITICAL: Log the model with proper artifacts
            logger.info("Logging model to MLflow...")
            
            # Log the sklearn model
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="lgbm_model",  # This creates the lgbm_model directory
                signature=signature,
                input_example=X_sample_dense[:1],  # Just one example
                registered_model_name=None  # Don't register during logging
            )
            
            logger.info(f"Model logged successfully to artifact path: lgbm_model")
            logger.info(f"Model info: {model_info}")

            # Log the vectorizer as a separate artifact
            logger.info("Logging vectorizer to MLflow...")
            mlflow.log_artifact(vectorizer_pkl_path, "vectorizer")
            
            # Save model info for registration
            model_path = "lgbm_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')
            logger.info(f"Model info saved for run: {run.info.run_id}")

            # Evaluate model and get metrics
            logger.info("Evaluating model...")
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log metrics properly
            logger.info("Logging metrics...")
            for label, metrics in report.items():
                if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:
                    # Log individual class metrics
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float, np.integer, np.floating)):
                            metric_key = f"test_class_{label}_{metric_name}"
                            mlflow.log_metric(metric_key, float(metric_value))

            # Log aggregate metrics
            if 'accuracy' in report:
                mlflow.log_metric("test_accuracy", float(report['accuracy']))
            
            if 'macro avg' in report:
                for metric_name, metric_value in report['macro avg'].items():
                    if isinstance(metric_value, (int, float, np.integer, np.floating)):
                        mlflow.log_metric(f"test_macro_{metric_name}", float(metric_value))
            
            if 'weighted avg' in report:
                for metric_name, metric_value in report['weighted avg'].items():
                    if isinstance(metric_value, (int, float, np.integer, np.floating)):
                        mlflow.log_metric(f"test_weighted_{metric_name}", float(metric_value))

            # Log confusion matrix
            log_confusion_matrix(cm, "Test_Data")

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")
            mlflow.set_tag("model_framework", "sklearn")

            # Verify that the model was logged
            client = mlflow.tracking.MlflowClient()
            artifacts = client.list_artifacts(run.info.run_id)
            logger.info(f"Artifacts in run: {[art.path for art in artifacts]}")
            
            # Check specifically for the model
            model_artifacts = client.list_artifacts(run.info.run_id, "lgbm_model")
            logger.info(f"Model artifacts: {[art.path for art in model_artifacts]}")

            logger.info(f"Model evaluation completed successfully. Run ID: {run.info.run_id}")
            print(f"✅ Model evaluation completed successfully!")
            print(f"📊 Run ID: {run.info.run_id}")
            print(f"🔗 Model logged to: runs:/{run.info.run_id}/lgbm_model")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"❌ Error: {e}")
            raise

if __name__ == '__main__':
    main()