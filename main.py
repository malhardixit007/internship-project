# main.py
# ✅ This is the ONLY file you need to run
# DON'T run src files individually (except predict.py)

from src.data_preprocessing import load_and_preprocess
from src.model import train_all_models, save_best_model
from src.evaluate import (plot_actual_vs_predicted, plot_model_comparison,
                           plot_feature_importance, plot_error_distribution)

print("=" * 50)
print("   SALES FORECASTING — ML PIPELINE")
print("=" * 50)

# Step 1: Load and preprocess
X_train, X_test, y_train, y_test = load_and_preprocess('data/sales_data.csv')

# Step 2: Train all models and compare
best_name, results = train_all_models(
    X_train, y_train, X_test, y_test
)

# Step 3: Save best model
best_model = results[best_name]['model']
save_best_model(best_model)

# Step 4: Generate all plots
plot_actual_vs_predicted(y_test, results)
plot_model_comparison(results)
plot_feature_importance(best_model, list(X_train.columns), best_name)
plot_error_distribution(y_test, results[best_name]['y_pred'], best_name)

print("\n" + "=" * 50)
print(f"DONE! Best model: {best_name}")
print("Check output_images/ for all graphs")
print("Check logs/training_log.txt for details")
print("=" * 50)