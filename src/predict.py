# src/predict.py
# Use this to make predictions with the saved model

import joblib
import pandas as pd

def predict(input_dict, model_path='models/best_model.pkl'):
    """
    Pass feature values as a dictionary and get predicted sales.
    Make sure the features match exactly what was used in training.
    """
    model = joblib.load(model_path)
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]
    print(f"\nPredicted Sales: {int(prediction)} units")
    return prediction

if __name__ == '__main__':
    # Example prediction for a Monday in June, no holiday
    sample = {
        'price': 35.0, 'discount': 0.10, 'holiday': 0,
        'temperature': 28.5, 'day_of_week': 0, 'month': 6,
        'quarter': 2, 'day_of_year': 160, 'week_of_year': 23,
        'is_weekend': 0, 'is_month_end': 0,
        'lag_1': 145, 'lag_7': 138, 'lag_30': 130,
        'rolling_mean_7': 142.0, 'rolling_mean_30': 135.0,
        'rolling_std_7': 8.5,
        'store_size_enc': 1, 'category_enc': 0, 'weekday_enc': 0
    }
    predict(sample)