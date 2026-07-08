# src/data_preprocessing.py
# DO    : clean data, encode categories, engineer features
# DON'T : hardcode column names that might change, don't drop data blindly

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    """
    Load raw CSV, engineer features, encode categoricals,
    and return train/test splits ready for ML.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns.")

    # ── Date features ──────────────────────────────────────────
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week']  = df['date'].dt.dayofweek       # 0=Mon, 6=Sun
    df['month']        = df['date'].dt.month
    df['quarter']      = df['date'].dt.quarter
    df['day_of_year']  = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend']   = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # ── Lag features (previous days' sales) ────────────────────
    # These teach the model that yesterday's sales affect today's
    df = df.sort_values('date').reset_index(drop=True)
    df['lag_1']  = df['sales'].shift(1)   # yesterday
    df['lag_7']  = df['sales'].shift(7)   # last week same day
    df['lag_30'] = df['sales'].shift(30)  # last month

    # ── Rolling averages ───────────────────────────────────────
    df['rolling_mean_7']  = df['sales'].shift(1).rolling(7).mean()
    df['rolling_mean_30'] = df['sales'].shift(1).rolling(30).mean()
    df['rolling_std_7']   = df['sales'].shift(1).rolling(7).std()

    # ── Encode categorical columns ─────────────────────────────
    le = LabelEncoder()
    df['store_size_enc'] = le.fit_transform(df['store_size'])
    df['category_enc']   = le.fit_transform(df['category'])
    df['weekday_enc']    = le.fit_transform(df['weekday'])

    # ── Drop columns not needed for ML ─────────────────────────
    df.drop(columns=['date', 'store_size', 'category', 'weekday'], inplace=True)

    # ── Drop rows with NaN (created by lag/rolling) ────────────
    df.dropna(inplace=True)
    print(f"After feature engineering: {len(df)} rows, {df.shape[1]} columns.")

    # ── Split features and target ──────────────────────────────
    X = df.drop(columns=['sales'])
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  
        # shuffle=False keeps time order intact — important for time series!
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test