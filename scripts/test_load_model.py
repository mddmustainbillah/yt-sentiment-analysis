# scripts/test_load_model.py
import mlflow.pyfunc
import pytest
import os
from mlflow.tracking import MlflowClient

# Set local MLflow tracking URI (matching your setup)
mlflow.set_tracking_uri("/Users/mustainbillah/Desktop/yt-sentiment-analysis/mlruns")

@pytest.mark.parametrize("model_name, stage", [
    ("yt_chrome_plugin_model", "staging"),
])
def test_load_latest_staging_model(model_name, stage):
    client = MlflowClient()
    
    try:
        # Try to get model using alias first (MLflow 2.0+) - matching your register_model.py approach
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
            latest_version = latest_version_info[0].version
            model_uri = f"models:/{model_name}/{latest_version}"
            print(f"Using version-based model URI: {model_uri}")
        
        # Load the model
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Ensure the model loads successfully
        assert model is not None, "Model failed to load"
        
        # Additional verification - check if model has predict method
        assert hasattr(model, 'predict'), "Model does not have predict method"
        
        print(f"✅ Model '{model_name}' loaded successfully from '{stage}' stage.")
        
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")

if __name__ == "__main__":
    test_load_latest_staging_model("yt_chrome_plugin_model", "staging")