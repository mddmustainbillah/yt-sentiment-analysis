# src/model/register_model.py

import json
import mlflow
import logging
import os
import time
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Set up MLflow tracking URI
mlflow.set_tracking_uri("/Users/mustainbillah/Desktop/yt-sentiment-analysis/mlruns")

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model info file not found: {file_path}")
            
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        
        # Validate required keys
        required_keys = ['run_id', 'model_path']
        for key in required_keys:
            if key not in model_info:
                raise ValueError(f"Missing required key '{key}' in model info")
        
        logger.debug('Model info loaded from %s: %s', file_path, model_info)
        return model_info
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except json.JSONDecodeError as e:
        logger.error('Invalid JSON in file %s: %s', file_path, e)
        raise
    except Exception as e:
        logger.error('Unexpected error loading model info: %s', e)
        raise

def verify_model_artifacts(run_id: str, model_path: str) -> bool:
    """Verify that the model artifacts exist in the MLflow run."""
    try:
        client = MlflowClient()
        
        # Check if run exists
        try:
            run = client.get_run(run_id)
            logger.info(f"Run {run_id} found with status: {run.info.status}")
        except Exception as e:
            logger.error(f"Run {run_id} not found: {e}")
            return False
        
        # List all artifacts in the run
        try:
            all_artifacts = client.list_artifacts(run_id)
            logger.info(f"All artifacts in run: {[art.path for art in all_artifacts]}")
        except Exception as e:
            logger.warning(f"Could not list all artifacts: {e}")
        
        # Check if model path exists
        try:
            model_artifacts = client.list_artifacts(run_id, model_path)
            if not model_artifacts:
                logger.error(f"No artifacts found at path '{model_path}' in run {run_id}")
                return False
            
            logger.info(f"Model artifacts found at '{model_path}': {[art.path for art in model_artifacts]}")
            
            # Check for essential MLflow model files
            artifact_paths = [art.path for art in model_artifacts]
            essential_files = ['MLmodel']  # MLmodel is the most critical file
            
            missing_files = []
            for file in essential_files:
                expected_path = f"{model_path}/{file}"
                if not any(expected_path in path for path in artifact_paths):
                    missing_files.append(file)
            
            if missing_files:
                logger.error(f"Missing essential model files: {missing_files}")
                return False
            
            logger.info("All essential model artifacts verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error checking model artifacts at '{model_path}': {e}")
            return False
    
    except Exception as e:
        logger.error(f"Error verifying model artifacts: {e}")
        return False

def register_model(model_name: str, model_info: dict) -> dict:
    """Register the model to the MLflow Model Registry."""
    try:
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        
        logger.info(f"Starting model registration for '{model_name}'")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Model path: {model_path}")
        
        # Verify artifacts exist before attempting registration
        if not verify_model_artifacts(run_id, model_path):
            raise ValueError(f"Model artifacts verification failed for run {run_id} at path {model_path}")
        
        # Construct the model URI
        model_uri = f"runs:/{run_id}/{model_path}"
        logger.info(f"Model URI: {model_uri}")
        
        # Add a delay to ensure the run is fully committed
        logger.info("Waiting for run to be fully committed...")
        time.sleep(3)
        
        # Register the model
        logger.info(f"Registering model '{model_name}' from URI: {model_uri}")
        
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            logger.info(f"✅ Successfully registered model '{model_name}' version {model_version.version}")
            
        except MlflowException as e:
            logger.error(f"MLflow exception during registration: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during model registration: {e}")
            raise
        
        # Set up model alias/stage
        client = MlflowClient()
        
        # Try the new alias system first (MLflow 2.0+)
        try:
            client.set_registered_model_alias(model_name, "staging", model_version.version)
            logger.info(f"✅ Set alias 'staging' for model {model_name} version {model_version.version}")
            stage_method = "alias"
        except Exception as alias_error:
            logger.warning(f"Could not set alias 'staging': {alias_error}")
            
            # Fallback to deprecated stage system
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Staging"
                )
                logger.info(f"✅ Set stage 'Staging' for model {model_name} version {model_version.version}")
                stage_method = "stage"
            except Exception as stage_error:
                logger.warning(f"Could not set stage 'Staging': {stage_error}")
                stage_method = "none"
        
        # Add model description
        try:
            client.update_registered_model(
                name=model_name,
                description="YouTube Comment Sentiment Analysis Model - LightGBM classifier trained on preprocessed comment data with TF-IDF features"
            )
            logger.info(f"✅ Updated model description for {model_name}")
        except Exception as e:
            logger.warning(f"Could not update model description: {e}")
        
        # Add version description
        try:
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=f"Model trained and registered via DVC pipeline. Run ID: {run_id}"
            )
            logger.info(f"✅ Updated version description for {model_name} v{model_version.version}")
        except Exception as e:
            logger.warning(f"Could not update version description: {e}")
        
        # Return registration summary
        return {
            "model_name": model_name,
            "version": model_version.version,
            "run_id": run_id,
            "model_uri": model_uri,
            "stage_method": stage_method,
            "registration_status": "success"
        }
        
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        print("🚀 Starting model registration process...")
        
        # Load model info
        model_info_path = 'experiment_info.json'
        logger.info(f"Loading model info from: {model_info_path}")
        
        if not os.path.exists(model_info_path):
            error_msg = f"Model info file not found: {model_info_path}"
            logger.error(error_msg)
            print(f"❌ Error: {error_msg}")
            print("💡 Make sure you've run the model evaluation step first!")
            return
        
        model_info = load_model_info(model_info_path)
        print(f"📋 Model info loaded: {model_info}")
        
        # Register model
        model_name = "yt_chrome_plugin_model"
        registration_result = register_model(model_name, model_info)
        
        # Print success summary
        print("\n" + "="*60)
        print("🎉 MODEL REGISTRATION SUCCESSFUL!")
        print("="*60)
        print(f"📝 Model Name: {registration_result['model_name']}")
        print(f"🔢 Version: {registration_result['version']}")
        print(f"🆔 Run ID: {registration_result['run_id']}")
        print(f"🔗 Model URI: {registration_result['model_uri']}")
        print(f"🏷️  Stage Method: {registration_result['stage_method']}")
        print("="*60)
        
        # Instructions for next steps
        print("\n📋 Next Steps:")
        print(f"1. View your model in MLflow UI: http://localhost:5000")
        print(f"2. Load model for inference using:")
        print(f"   model = mlflow.sklearn.load_model('models:/{model_name}/staging')")
        print(f"3. Or use the specific version:")
        print(f"   model = mlflow.sklearn.load_model('models:/{model_name}/{registration_result['version']}')")
        
        logger.info("Model registration completed successfully")
        
    except Exception as e:
        logger.error('Failed to complete model registration: %s', e)
        print(f"❌ Registration failed: {e}")
        
        # Provide troubleshooting tips
        print("\n🔧 Troubleshooting Tips:")
        print("1. Ensure model evaluation completed successfully")
        print("2. Check if experiment_info.json exists and is valid")
        print("3. Verify MLflow tracking URI is correct")
        print("4. Check MLflow server is running if using remote tracking")

if __name__ == '__main__':
    main()