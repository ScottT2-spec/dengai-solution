"""
DengAI: Predicting Disease Spread — Top Solution
Run in Google Colab. Upload the 3 CSV files to Google Drive first.

Approach:
- Separate models per city (San Juan vs Iquitos)
- Heavy feature engineering (lags, rolling stats, seasonal)
- Ensemble: XGBoost + LightGBM + CatBoost
- Time-series cross-validation
- Post-process: round to non-negative integers
- Metric: Mean Absolute Error (MAE)
"""

# ============================================================
# CELL 1: Setup
# ============================================================

# !pip install -q xgboost lightgbm catboost scikit-learn pandas numpy

# from google.colab import drive
# drive.mount('/content/drive')

# UPDATE these paths to where you put the CSV files
DATA_DIR = "/content/drive/MyDrive/dengai"  # or wherever you upload them
# If running locally for testing:
# DATA_DIR = "."

# ============================================================
# CELL 2: Load data
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

train_features = pd.read_csv(f"{DATA_DIR}/dengue_features_train.csv")
train_labels = pd.read_csv(f"{DATA_DIR}/dengue_labels_train.csv")
test_features = pd.read_csv(f"{DATA_DIR}/dengue_features_test.csv")

# Merge features and labels
train = train_features.merge(train_labels, on=["city", "year", "weekofyear"])

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test_features.shape}")
print(f"Cities: {train.city.unique()}")
print(f"\nSan Juan: {len(train[train.city=='sj'])} weeks")
print(f"Iquitos:  {len(train[train.city=='iq'])} weeks")
print(f"\nTarget stats:")
print(train.groupby("city")["total_cases"].describe())

# ============================================================
# CELL 3: Feature engineering
# ============================================================

def engineer_features(df, is_train=True):
    """Create features from the raw data. Works for both train and test."""
    df = df.copy()

    # Drop week_start_date (string, not useful as-is)
    if "week_start_date" in df.columns:
        # Extract month first
        df["month"] = pd.to_datetime(df["week_start_date"]).dt.month
        df.drop("week_start_date", axis=1, inplace=True)
    
    # Season (tropical wet/dry)
    df["is_wet_season"] = df["month"].apply(lambda m: 1 if m in [5,6,7,8,9,10,11] else 0)
    
    # Cyclical encoding of week
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    
    # Cyclical encoding of month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Fill NaNs with forward fill per city, then backfill remaining
    climate_cols = [c for c in df.columns if c not in ["city", "year", "weekofyear", "total_cases", "month",
                                                         "is_wet_season", "week_sin", "week_cos", "month_sin", "month_cos"]]
    
    for city in df.city.unique():
        mask = df.city == city
        df.loc[mask, climate_cols] = df.loc[mask, climate_cols].ffill().bfill()

    # Average NDVI
    ndvi_cols = [c for c in df.columns if "ndvi" in c]
    df["ndvi_avg"] = df[ndvi_cols].mean(axis=1)

    # Temperature range features
    if "reanalysis_max_air_temp_k" in df.columns and "reanalysis_min_air_temp_k" in df.columns:
        df["temp_range_k"] = df["reanalysis_max_air_temp_k"] - df["reanalysis_min_air_temp_k"]
    
    if "station_max_temp_c" in df.columns and "station_min_temp_c" in df.columns:
        df["station_temp_range"] = df["station_max_temp_c"] - df["station_min_temp_c"]

    # Humidity * temperature interaction (mosquito breeding conditions)
    if "reanalysis_specific_humidity_g_per_kg" in df.columns and "reanalysis_avg_temp_k" in df.columns:
        df["humidity_temp_interaction"] = df["reanalysis_specific_humidity_g_per_kg"] * df["reanalysis_avg_temp_k"]

    # Precipitation features
    precip_cols = [c for c in df.columns if "precip" in c]
    if len(precip_cols) > 0:
        df["precip_avg"] = df[precip_cols].mean(axis=1)

    # --- LAG FEATURES (per city) ---
    lag_cols = [
        "reanalysis_specific_humidity_g_per_kg",
        "reanalysis_dew_point_temp_k",
        "reanalysis_avg_temp_k",
        "station_avg_temp_c",
        "precipitation_amt_mm",
        "reanalysis_precip_amt_kg_per_m2",
        "ndvi_avg",
        "humidity_temp_interaction",
    ]
    
    # Only use columns that exist
    lag_cols = [c for c in lag_cols if c in df.columns]

    for city in df.city.unique():
        mask = df.city == city
        city_data = df.loc[mask].copy()
        
        for col in lag_cols:
            # Lags: 1-4 weeks (mosquito incubation ~1-2 weeks, disease incubation ~1 week)
            for lag in [1, 2, 3, 4]:
                df.loc[mask, f"{col}_lag{lag}"] = city_data[col].shift(lag)
            
            # Rolling means: 4, 8, 12 weeks
            for window in [4, 8, 12]:
                df.loc[mask, f"{col}_roll{window}"] = city_data[col].rolling(window, min_periods=1).mean()
            
            # Rolling std (variability)
            df.loc[mask, f"{col}_std4"] = city_data[col].rolling(4, min_periods=1).std()
        
        # Target lags (only for train — we'll handle test separately)
        if "total_cases" in city_data.columns:
            for lag in [1, 2, 3, 4]:
                df.loc[mask, f"cases_lag{lag}"] = city_data["total_cases"].shift(lag)
            df.loc[mask, "cases_roll4"] = city_data["total_cases"].rolling(4, min_periods=1).mean()
            df.loc[mask, "cases_roll8"] = city_data["total_cases"].rolling(8, min_periods=1).mean()

    # Fill any remaining NaN from lag/rolling
    df = df.ffill().bfill()
    
    # Fill any final NaN with 0
    df = df.fillna(0)

    return df


print("Engineering features...")
train_eng = engineer_features(train, is_train=True)
test_eng = engineer_features(test_features, is_train=False)
print(f"Train features: {train_eng.shape[1]} columns")
print(f"Test features:  {test_eng.shape[1]} columns")

# ============================================================
# CELL 4: Prepare for modeling
# ============================================================

# Columns to drop before modeling
drop_cols = ["city", "total_cases"]
feature_cols = [c for c in train_eng.columns if c not in drop_cols]

# Make sure test has same features (minus target lag features)
# For test, target lag features won't exist — fill with 0
for col in feature_cols:
    if col not in test_eng.columns:
        test_eng[col] = 0

test_eng = test_eng[feature_cols]  # same column order

print(f"Feature count: {len(feature_cols)}")

# ============================================================
# CELL 5: Train models per city
# ============================================================

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def get_models():
    """Return a dict of models to ensemble."""
    return {
        "xgb": XGBRegressor(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=42,
            verbosity=0,
        ),
        "lgbm": LGBMRegressor(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=10,
            random_state=42,
            verbose=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=1000,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        ),
    }


def train_and_predict_city(city_code, train_df, test_df, feature_cols):
    """Train ensemble for one city, return test predictions."""
    city_train = train_df[train_df.city == city_code].copy()
    city_test = test_df[test_df.city == city_code].copy() if "city" in test_df.columns else test_df.copy()
    
    X_train = city_train[feature_cols].values
    y_train = city_train["total_cases"].values
    X_test = city_test[feature_cols].values if len(city_test) > 0 else None
    
    print(f"\n{'='*50}")
    print(f"City: {city_code.upper()} | Train: {len(X_train)} | Test: {len(X_test) if X_test is not None else 0}")
    print(f"{'='*50}")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = get_models()
    trained_models = {}
    cv_scores = {}
    
    for name, model in models.items():
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model_clone = model.__class__(**model.get_params())
            
            if name == "xgb":
                model_clone.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                              verbose=False)
            elif name == "lgbm":
                model_clone.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                              callbacks=[])
            else:
                model_clone.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                              early_stopping_rounds=50)
            
            val_pred = model_clone.predict(X_val)
            val_pred = np.clip(np.round(val_pred), 0, None)
            mae = mean_absolute_error(y_val, val_pred)
            fold_scores.append(mae)
        
        avg_mae = np.mean(fold_scores)
        cv_scores[name] = avg_mae
        print(f"  {name:10s} CV MAE: {avg_mae:.2f} (±{np.std(fold_scores):.2f})")
        
        # Retrain on full data
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    # Ensemble prediction (weighted average — better models get more weight)
    if X_test is not None and len(X_test) > 0:
        # Inverse MAE weighting
        total_inv_mae = sum(1.0 / s for s in cv_scores.values())
        weights = {name: (1.0 / score) / total_inv_mae for name, score in cv_scores.items()}
        
        print(f"\n  Ensemble weights: {', '.join(f'{n}={w:.3f}' for n,w in weights.items())}")
        
        test_pred = np.zeros(len(X_test))
        for name, model in trained_models.items():
            pred = model.predict(X_test)
            test_pred += weights[name] * pred
        
        # Round to non-negative integers
        test_pred = np.clip(np.round(test_pred), 0, None).astype(int)
        
        best_cv = min(cv_scores.values())
        print(f"  Best single model CV MAE: {best_cv:.2f}")
        
        return test_pred
    
    return None


# Need test features with city column for splitting
test_with_city = test_features[["city"]].copy()
test_with_city = pd.concat([test_with_city, test_eng], axis=1)

# Train and predict for each city
sj_predictions = train_and_predict_city("sj", train_eng, test_with_city, feature_cols)
iq_predictions = train_and_predict_city("iq", train_eng, test_with_city, feature_cols)

# ============================================================
# CELL 6: Multi-seed ensemble (extra boost)
# ============================================================

print("\n\nRunning multi-seed ensemble for robustness...")

all_sj_preds = [sj_predictions]
all_iq_preds = [iq_predictions]

for seed in [123, 456, 789]:
    # Quick retrain with different seeds
    for city_code, pred_list in [("sj", all_sj_preds), ("iq", all_iq_preds)]:
        city_train = train_eng[train_eng.city == city_code]
        city_test = test_with_city[test_with_city.city == city_code]
        
        X_train = city_train[feature_cols].values
        y_train = city_train["total_cases"].values
        X_test = city_test[feature_cols].values
        
        preds = np.zeros(len(X_test))
        
        for Model, params in [
            (XGBRegressor, dict(n_estimators=1000, max_depth=5, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.7, random_state=seed, verbosity=0)),
            (LGBMRegressor, dict(n_estimators=1000, max_depth=5, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.7, random_state=seed, verbose=-1)),
            (CatBoostRegressor, dict(iterations=1000, depth=5, learning_rate=0.03,
                                    random_seed=seed, verbose=0)),
        ]:
            m = Model(**params)
            m.fit(X_train, y_train)
            preds += m.predict(X_test) / 3
        
        pred_list.append(np.clip(np.round(preds), 0, None).astype(int))

# Average across all seeds
sj_final = np.round(np.mean(all_sj_preds, axis=0)).astype(int)
iq_final = np.round(np.mean(all_iq_preds, axis=0)).astype(int)

print(f"San Juan predictions: min={sj_final.min()}, max={sj_final.max()}, mean={sj_final.mean():.1f}")
print(f"Iquitos predictions:  min={iq_final.min()}, max={iq_final.max()}, mean={iq_final.mean():.1f}")

# ============================================================
# CELL 7: Create submission
# ============================================================

# Build submission DataFrame
sj_test = test_features[test_features.city == "sj"][["city", "year", "weekofyear"]].copy()
iq_test = test_features[test_features.city == "iq"][["city", "year", "weekofyear"]].copy()

sj_test["total_cases"] = sj_final
iq_test["total_cases"] = iq_final

submission = pd.concat([sj_test, iq_test], ignore_index=True)

# Save
output_path = f"{DATA_DIR}/submission.csv"
submission.to_csv(output_path, index=False)

print(f"\nSubmission saved to: {output_path}")
print(f"Shape: {submission.shape}")
print(f"\nFirst few rows:")
print(submission.head(10))
print(f"\nLast few rows:")
print(submission.tail(10))

print("\n=== DONE! ===")
print(f"Upload {output_path} to DrivenData:")
print("https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/submissions/")
