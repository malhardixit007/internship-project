# src/evaluate.py
# DO    : save ALL plots to output_images/ with clear names
# DON'T : use plt.show() alone without saving first

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('output_images', exist_ok=True)

def plot_actual_vs_predicted(y_test, results):
    """Line chart comparing all models against actual sales"""
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.values[:90], label='Actual Sales',
             color='black', linewidth=2)

    colors = ['blue', 'orange', 'green', 'red']
    for (name, res), color in zip(results.items(), colors):
        plt.plot(res['y_pred'][:90], label=f'{name} (MAE={res["MAE"]})',
                 linestyle='--', alpha=0.75, color=color)

    plt.title('Actual vs Predicted Sales — All Models (First 90 Days)')
    plt.xlabel('Day Index')
    plt.ylabel('Sales Units')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_images/actual_vs_predicted.png', dpi=150)
    plt.close()
    print("Saved: actual_vs_predicted.png")

def plot_model_comparison(results):
    """Bar chart comparing MAE of all models"""
    names = list(results.keys())
    maes  = [results[n]['MAE'] for n in names]
    r2s   = [results[n]['R2']  for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(names, maes, color=['#4C72B0','#DD8452','#55A868','#C44E52'])
    axes[0].set_title('Model Comparison — MAE (lower is better)')
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].bar(names, r2s, color=['#4C72B0','#DD8452','#55A868','#C44E52'])
    axes[1].set_title('Model Comparison — R² Score (higher is better)')
    axes[1].set_ylabel('R² Score')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig('output_images/model_comparison.png', dpi=150)
    plt.close()
    print("Saved: model_comparison.png")

def plot_feature_importance(model, feature_names, model_name):
    """Feature importance for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        print(f"Skipping feature importance for {model_name} (not tree-based)")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # top 15 features

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)),
            importances[indices], color='steelblue')
    plt.xticks(range(len(indices)),
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title(f'Top Feature Importances — {model_name}')
    plt.tight_layout()
    plt.savefig('output_images/feature_importance.png', dpi=150)
    plt.close()
    print("Saved: feature_importance.png")

def plot_error_distribution(y_test, y_pred, model_name):
    """Histogram of prediction errors"""
    errors = y_test.values - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(errors, bins=30, kde=True, color='coral')
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f'Prediction Error Distribution — {model_name}')
    plt.xlabel('Error (Actual - Predicted)')
    plt.tight_layout()
    plt.savefig('output_images/error_distribution.png', dpi=150)
    plt.close()
    print("Saved: error_distribution.png")