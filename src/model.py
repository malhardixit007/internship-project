# src/model.py
# DO    : try multiple models, compare them, save the best
# DON'T : only train one model and assume it's best

import joblib
import os
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

MODELS = {
    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1        # use all CPU cores
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
}

def train_all_models(X_train, y_train, X_test, y_test, log_path='logs/training_log.txt'):
    """Train all models, log results, return best model name + all results"""
    os.makedirs('logs', exist_ok=True)
    results = {}

    with open(log_path, 'w') as log:
        log.write("=== SALES FORECASTING TRAINING LOG ===\n\n")

        for name, model in MODELS.items():
            print(f"\nTraining: {name}...")
            start = time.time()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
            r2   = r2_score(y_test, y_pred)
            elapsed = time.time() - start

            results[name] = {
                'model' : model,
                'y_pred': y_pred,
                'MAE'   : round(mae, 2),
                'RMSE'  : round(rmse, 2),
                'R2'    : round(r2, 4)
            }

            line = f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}, Time={elapsed:.1f}s"
            print(line)
            log.write(line + "\n")

    # Pick best model by lowest MAE
    best_name = min(results, key=lambda k: results[k]['MAE'])
    print(f"\n✅ Best Model: {best_name} (MAE={results[best_name]['MAE']})")

    return best_name, results

def save_best_model(model, path='models/best_model.pkl'):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {path}")