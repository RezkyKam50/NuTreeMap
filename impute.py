from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt, pandas as pd

def impute_with_metrics(df_train, df_full, target_col='saturated_fat'):
    features = [
        'total_fat', 'monounsaturated_fatty_acids',
        'polyunsaturated_fatty_acids', 'fatty_acids_total_trans',
        'cholesterol'
    ]

    known = df_train
    X = known[features]
    y = known[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBRegressor(n_estimators=1000, eval_metric="rmse", verbosity=0, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Imputation Model Performance Metrics:")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
    print(f"R²:   {r2_score(y_test, y_pred):.4f}")

    # Predict missing values in full dataset
    missing_mask = df_full[target_col].isna()
    X_missing = df_full.loc[missing_mask, features]
    predicted = model.predict(X_missing)
    # Impute
    df_full.loc[missing_mask, target_col] = predicted
    print(f"Imputed {len(predicted)} missing values.")

    # Save
    df_full.to_parquet("nutritional_complete_imputed.parquet", index=False)

    print(len(df_full)) # 8789
    print("Saved final imputed DataFrame.")

# Load data
df_to_be_imputed = pd.read_parquet("nutritional_complete.parquet")        # full set (8789 rows)
df_to_be_feature_extracted = pd.read_parquet("nutritional_complete_DROPNA.parquet")  # clean training set (7199 rows)

# Run
impute_with_metrics(df_to_be_feature_extracted, df_to_be_imputed)

'''
MAE:  0.5295
RMSE: 1.4769
R²:   0.9607
Imputed 1590 missing values.
8789
'''