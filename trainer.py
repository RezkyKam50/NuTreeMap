import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from scipy.sparse import hstack
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, pickle, joblib, os, datetime
warnings.filterwarnings('ignore')

def save_models_and_components(model1, model2, vectorizer,
                              target_features, all_nutrients, input_features, save_dir="nutrition_models"):

    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving models to directory: {save_dir}")
   
    joblib.dump(model1, os.path.join(save_dir, "model1_text_numeric.joblib"))
    joblib.dump(model2, os.path.join(save_dir, "model2_text_only.joblib"))
    
    joblib.dump(vectorizer, os.path.join(save_dir, "tfidf_vectorizer.joblib"))

    metadata = {
        'target_features': target_features,
        'all_nutrients': all_nutrients,
        'input_features': input_features,
        'save_date': datetime.datetime.now().isoformat(),
        'model_info': {
            'model1_type': 'RegressorChain with XGBoost',
            'model2_type': 'RegressorChain with XGBoost',
            'model1_description': 'Text + Numeric Features -> Subset of Nutrients',
            'model2_description': 'Text Features Only -> All Nutrients'
        }
    }
    
    with open(os.path.join(save_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print("✓ Model 1 (Text + Numeric) saved")
    print("✓ Model 2 (Text Only) saved")
    print("✓ TF-IDF Vectorizer saved")
    print("✓ Metadata saved")
    print(f"All components saved successfully to '{save_dir}' directory!")


def multioutput_r2_scorer(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(r2_score(y_true[:, i], y_pred[:, i]))
    return np.mean(scores)

def multioutput_mae_scorer(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(mean_absolute_error(y_true[:, i], y_pred[:, i]))
    return np.mean(scores)

# Load dataset
print("Loading dataset...")
df = pd.read_parquet("umapped3D.parquet")
df = df[['protein', 'calories', 'calcium', 'total_fat', 'saturated_fat', 'cholesterol', 'sodium',
         'carbohydrate', 'fat', 'fiber', 'saturated_fatty_acids', 'monounsaturated_fatty_acids',
         'polyunsaturated_fatty_acids', 'fatty_acids_total_trans', 'name', 'food_type_1', 'food_type_2']]

print("Creating text features...")
df['combined_text'] = df['name'].astype(str) + ', ' + df['food_type_1'].fillna('').astype(str) + ', ' + df['food_type_2'].fillna('').astype(str)
vectorizer = TfidfVectorizer(max_features=10000)
food_name_features = vectorizer.fit_transform(df['combined_text'])

input_features = ['protein', 'total_fat', 'carbohydrate', 'sodium', 'cholesterol']
target_features = ['calories', 'calcium', 'saturated_fat', 'fiber', 'saturated_fatty_acids', 
                   'monounsaturated_fatty_acids', 'polyunsaturated_fatty_acids', 'fatty_acids_total_trans']

all_nutrients = ['protein', 'calories', 'calcium', 'total_fat', 'saturated_fat', 'cholesterol', 
                'sodium', 'carbohydrate', 'fiber', 'saturated_fatty_acids', 
                'monounsaturated_fatty_acids', 'polyunsaturated_fatty_acids', 
                'fatty_acids_total_trans']

print("Preparing Model 1 data (Text + Numeric -> Subset)...")
X_numeric = df[input_features].values
X_combined = hstack([food_name_features, X_numeric])
y_subset = df[target_features].values

print("Preparing Model 2 data (Text Only -> All Nutrients)...")
X_text_only = food_name_features
y_all = df[all_nutrients].values

cv_folds = 5
kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

# Create models
base_model = XGBRegressor(
    objective='reg:squarederror', 
    random_state=42, 
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

model1 = MultiOutputRegressor(base_model)
model2 = MultiOutputRegressor(base_model)

r2_scorer = make_scorer(multioutput_r2_scorer, greater_is_better=True)
mae_scorer = make_scorer(multioutput_mae_scorer, greater_is_better=False)

X_combined_dense = X_combined.toarray()

cv_r2_scores_1 = cross_val_score(model1, X_combined_dense, y_subset, cv=kfold, scoring=r2_scorer, n_jobs=-1)
cv_mae_scores_1 = cross_val_score(model1, X_combined_dense, y_subset, cv=kfold, scoring=mae_scorer, n_jobs=-1)


X_text_dense = X_text_only.toarray()

cv_r2_scores_2 = cross_val_score(model2, X_text_dense, y_all, cv=kfold, scoring=r2_scorer, n_jobs=-1)
cv_mae_scores_2 = cross_val_score(model2, X_text_dense, y_all, cv=kfold, scoring=mae_scorer, n_jobs=-1)


X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    X_combined_dense, y_subset, test_size=0.2, random_state=42)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_text_dense, y_all, test_size=0.2, random_state=42)

model1.fit(X_train_1, y_train_1)
y_pred_1 = model1.predict(X_test_1)
for i, target in enumerate(target_features):
    mae = mean_absolute_error(y_test_1[:, i], y_pred_1[:, i])
    mse = mean_squared_error(y_test_1[:, i], y_pred_1[:, i])
    r2 = r2_score(y_test_1[:, i], y_pred_1[:, i])
    print(f"{target:25} | MAE: {mae:7.2f} | MSE: {mse:10.2f} | R²: {r2:6.3f}")

model2.fit(X_train_2, y_train_2)
y_pred_2 = model2.predict(X_test_2)

for i, nutrient in enumerate(all_nutrients):
    mae = mean_absolute_error(y_test_2[:, i], y_pred_2[:, i])
    mse = mean_squared_error(y_test_2[:, i], y_pred_2[:, i])
    r2 = r2_score(y_test_2[:, i], y_pred_2[:, i])
    print(f"{nutrient:25} | MAE: {mae:7.2f} | MSE: {mse:10.2f} | R²: {r2:6.3f}")

new_food_text = "Chicken Salad, Salted"
new_food_features = vectorizer.transform([new_food_text])
 
new_numeric = np.array([[
    25.0,   # protein (g)
    3.0,    # total_fat (g)
    120.0,  # carbohydrate (g)
    20.0,   # sodium (mg)
    100.0   # cholesterol (mg)
]])

new_X_combined = hstack([new_food_features, new_numeric])
new_X_combined_dense = new_X_combined.toarray()
 
predicted_values_1 = model1.predict(new_X_combined_dense)[0]
print(f"\nModel 1 Predictions for '{new_food_text}':")
print("-" * 50)
for i, target in enumerate(target_features):
    print(f"{target:25}: {predicted_values_1[i]:8.2f}")
 
predicted_values_2 = model2.predict(new_food_features.toarray())[0]
for i, nutrient in enumerate(all_nutrients):
    print(f"{nutrient:25}: {predicted_values_2[i]:8.2f}")

save_models_and_components(
    model1=model1,
    model2=model2, 
    vectorizer=vectorizer,
    target_features=target_features,
    all_nutrients=all_nutrients,
    input_features=input_features
)
 
cv_df = pd.DataFrame({
    "Fold": list(range(1, cv_folds + 1)) * 2,
    "Model": ["Model 1"] * cv_folds + ["Model 2"] * cv_folds,
    "R2": np.concatenate([cv_r2_scores_1, cv_r2_scores_2]),
    "MAE": -np.concatenate([cv_mae_scores_1, cv_mae_scores_2])   
})
 
cv_summary = cv_df.groupby("Model").agg({
    "R2": ['mean', 'std'],
    "MAE": ['mean', 'std']
}).reset_index()
palette = sns.color_palette("pastel")
cv_summary.columns = ['Model', 'R2_mean', 'R2_std', 'MAE_mean', 'MAE_std']
 
fig, ax1 = plt.subplots(figsize=(12, 6))
 
x = np.arange(len(cv_summary))
bar_width = 0.35
bars1 = ax1.bar(x - bar_width/2, cv_summary["R2_mean"], width=bar_width, 
                yerr=cv_summary["R2_std"], capsize=5, label='R²', color="skyblue", alpha=0.7)
ax1.set_xlabel('Model')
ax1.set_ylabel('R² Score', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(x)
ax1.set_xticklabels(cv_summary["Model"])
ax1.grid(True, alpha=0.3)
 
for i, (bar, mean_val, std_val) in enumerate(zip(bars1, cv_summary["R2_mean"], cv_summary["R2_std"])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
             f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', color='blue', fontsize=9)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + bar_width/2, cv_summary["MAE_mean"], width=bar_width, 
                yerr=cv_summary["MAE_std"], capsize=5, label='MAE', color="lightcoral", alpha=0.7)
ax2.set_ylabel('MAE', color='red')
ax2.tick_params(axis='y', labelcolor='red')
 
for i, (bar, mean_val, std_val) in enumerate(zip(bars2, cv_summary["MAE_mean"], cv_summary["MAE_std"])):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + std_val + (max(cv_summary["MAE_mean"]) * 0.05),
             f'{mean_val:.2f}±{std_val:.2f}', ha='center', va='bottom', color='red', fontsize=9)
 
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("Cross-Validation Metrics Comparison\n(R² and MAE with Standard Deviations)", fontsize=14)
plt.tight_layout()
plt.savefig("./metricsplot/metrics_dual_axis.png", dpi=300, bbox_inches='tight')
 
model1_metrics = {
    "Target": [],
    "MAE": [],
    "R2": []
}

for i, target in enumerate(target_features):
    model1_metrics["Target"].append(target)
    model1_metrics["MAE"].append(mean_absolute_error(y_test_1[:, i], y_pred_1[:, i]))
    model1_metrics["R2"].append(r2_score(y_test_1[:, i], y_pred_1[:, i]))
df_model1 = pd.DataFrame(model1_metrics)
 
model2_metrics = {
    "Target": [],
    "MAE": [],
    "R2": []
}
for i, target in enumerate(all_nutrients):
    model2_metrics["Target"].append(target)
    model2_metrics["MAE"].append(mean_absolute_error(y_test_2[:, i], y_pred_2[:, i]))
    model2_metrics["R2"].append(r2_score(y_test_2[:, i], y_pred_2[:, i]))
df_model2 = pd.DataFrame(model2_metrics)
 
all_mae_values = list(df_model1["MAE"]) + list(df_model2["MAE"])
all_r2_values = list(df_model1["R2"]) + list(df_model2["R2"])

mae_min, mae_max = min(all_mae_values), max(all_mae_values)
r2_min, r2_max = min(all_r2_values), max(all_r2_values)
 
mae_padding = (mae_max - mae_min) * 0.1
r2_padding = (r2_max - r2_min) * 0.1

mae_xlim = (max(0, mae_min - mae_padding), mae_max + mae_padding)
r2_xlim = (r2_min - r2_padding, min(1.0, r2_max + r2_padding))
 
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
 
sns.barplot(data=df_model1, x="MAE", y="Target", ax=axes[0, 0], palette="Blues_d")
axes[0, 0].set_title("Model 1 - MAE per Target (Subset)")
axes[0, 0].set_xlim(mae_xlim)
 
sns.barplot(data=df_model1, x="R2", y="Target", ax=axes[0, 1], palette="Blues_d")
axes[0, 1].set_title("Model 1 - R² per Target (Subset)")
axes[0, 1].set_xlim(r2_xlim)
 
sns.barplot(data=df_model2, x="MAE", y="Target", ax=axes[1, 0], palette="Reds_d")
axes[1, 0].set_title("Model 2 - MAE per Target (All Nutrients)")
axes[1, 0].set_xlim(mae_xlim)
 
sns.barplot(data=df_model2, x="R2", y="Target", ax=axes[1, 1], palette="Reds_d")
axes[1, 1].set_title("Model 2 - R² per Target (All Nutrients)")
axes[1, 1].set_xlim(r2_xlim)
 
for ax in axes.flat:
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.suptitle("Individual Target Performance: Model 1 vs Model 2", fontsize=16, y=1.02)
plt.savefig("./metricsplot/individual_target_performance.png", dpi=300, bbox_inches='tight')


 