# scripts/promote_model.py
import os
import mlflow
import logging
from mlflow.tracking import MlflowClient

def promote_model():
    """Promote model from staging to production after successful tests."""
    
    # Set local MLflow tracking URI (matching your setup)
    mlflow.set_tracking_uri("/Users/mustainbillah/Desktop/yt-sentiment-analysis/mlruns")
    client = MlflowClient()
    model_name = "yt_chrome_plugin_model"
    
    # Set up logging (matching your register_model.py style)
    logger = logging.getLogger('model_promotion')
    logger.setLevel('DEBUG')
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    try:
        logger.info(f"Starting model promotion process for '{model_name}'")
        
        # Get the current staging model (matching your register_model.py logic)
        try:
            # Try alias-based approach first (MLflow 2.0+)
            staging_model = client.get_model_version_by_alias(model_name, "staging")
            staging_version = staging_model.version
            logger.info(f"Found staging model using alias: version {staging_version}")
            stage_method = "alias"
        except Exception as alias_error:
            logger.warning(f"Could not get staging model using alias: {alias_error}")
            # Fallback to stage-based approach
            try:
                staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
                if not staging_versions:
                    raise ValueError(f"No model found in staging for '{model_name}'")
                staging_version = staging_versions[0].version
                logger.info(f"Found staging model using stages: version {staging_version}")
                stage_method = "stage"
            except Exception as stage_error:
                logger.error(f"Could not get staging model using stages: {stage_error}")
                raise ValueError(f"No staging model found for '{model_name}'")
        
        # Archive current production models (if any)
        try:
            if stage_method == "alias":
                try:
                    prod_model = client.get_model_version_by_alias(model_name, "production")
                    # Delete the production alias to archive it
                    client.delete_registered_model_alias(model_name, "production")
                    logger.info(f"Archived previous production model version {prod_model.version}")
                except:
                    logger.info("No existing production model with alias to archive")
            else:
                # Stage-based archiving
                prod_versions = client.get_latest_versions(model_name, stages=["Production"])
                for version in prod_versions:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version.version,
                        stage="Archived"
                    )
                    logger.info(f"Archived production model version {version.version}")
        except Exception as e:
            logger.warning(f"Error during archiving (continuing): {e}")
        
        # Promote the staging model to production (matching register_model.py approach)
        try:
            if stage_method == "alias":
                # Try alias-based promotion (MLflow 2.0+)
                client.set_registered_model_alias(model_name, "production", staging_version)
                logger.info(f"✅ Model version {staging_version} promoted to production using alias")
            else:
                # Fallback to stage-based promotion
                client.transition_model_version_stage(
                    name=model_name,
                    version=staging_version,
                    stage="Production"
                )
                logger.info(f"✅ Model version {staging_version} promoted to production using stage")
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
        
        # Update model version description (matching register_model.py style)
        try:
            client.update_model_version(
                name=model_name,
                version=staging_version,
                description=f"Model promoted to production after passing all tests. Original staging version: {staging_version}"
            )
            logger.info(f"Updated description for production model version {staging_version}")
        except Exception as e:
            logger.warning(f"Could not update model description: {e}")
        
        # Add production tag (matching register_model.py approach)
        try:
            client.set_model_version_tag(
                name=model_name,
                version=staging_version,
                key="deployment_status",
                value="production"
            )
            logger.info(f"Added production tag to model version {staging_version}")
        except Exception as e:
            logger.warning(f"Could not add production tag: {e}")
        
        # Success output (matching your register_model.py style)
        print("\n" + "="*60)
        print("🎉 MODEL PROMOTION SUCCESSFUL!")
        print("="*60)
        print(f"📝 Model Name: {model_name}")
        print(f"🔢 Version: {staging_version}")
        print(f"🏷️  Stage Method: {stage_method}")
        print(f"📊 Model URI: models:/{model_name}/production")
        print("="*60)
        
        # Instructions for next steps (matching register_model.py style)
        print("\n📋 Next Steps:")
        print(f"1. View your model in MLflow UI: http://localhost:5000")
        print(f"2. Load production model for inference using:")
        print(f"   model = mlflow.sklearn.load_model('models:/{model_name}/production')")
        print(f"3. Or use the specific version:")
        print(f"   model = mlflow.sklearn.load_model('models:/{model_name}/{staging_version}')")
        
        logger.info("Model promotion completed successfully")
        
    except Exception as e:
        logger.error(f'Failed to promote model: {e}')
        print(f"❌ Promotion failed: {e}")
        
        # Provide troubleshooting tips (matching register_model.py style)
        print("\n🔧 Troubleshooting Tips:")
        print("1. Ensure model registration completed successfully")
        print("2. Check if staging model exists in MLflow")
        print("3. Verify MLflow tracking URI is correct")
        print("4. Check MLflow server is running if using remote tracking")
        raise

if __name__ == "__main__":
    promote_model()