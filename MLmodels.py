# models.py
import joblib
from config import MODEL_PATHS, EXPECTED_FEATURES

def load_models():
    """Load ML models and validate features"""
    try:
        print("\nLoading ML models...")
        
        # Load Random Forest for risk score prediction
        risk_model = joblib.load(MODEL_PATHS['random_forest'])
        print("Random Forest model loaded successfully")
        
        # Load Isolation Forest for transaction monitoring
        isolation_model = joblib.load(MODEL_PATHS['isolation_forest'])
        print("Isolation Forest model loaded successfully")
        
        # Validate Random Forest model features
        if hasattr(risk_model, 'feature_names_in_'):
            model_features = list(risk_model.feature_names_in_)
            if len(model_features) != len(EXPECTED_FEATURES):
                print(f"ERROR: Random Forest model expects {len(model_features)} features but we're providing {len(EXPECTED_FEATURES)}")
                return None, None
            else:
                print("Random Forest model feature count validated")
        else:
            print("Warning: Random Forest model does not have feature names, assuming correct order")
        
        return risk_model, isolation_model
        
    except Exception as e:
        print(f"Error loading ML models: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None, None