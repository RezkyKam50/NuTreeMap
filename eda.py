import matplotlib.pyplot as plt, pandas as pd, xgboost as xgb, joblib, math, pickle, os

df = pd.read_parquet("./umapped3D.parquet")

def plotisna():
    missing = df.isna().sum()
    print("Missing values per column:")
    print(missing['saturated_fat'])
    
    plt.figure(figsize=(12, 6))
    missing.plot(kind='bar', color='salmon')
    plt.title('Missing Values (NaN) After Imputation')
    plt.ylabel('Count of Missing Values')
    plt.xlabel('Column Name')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"./metricsplot/nan_saturated_imputed.png")
    print(len(df))
    print(df.head(10))
    plt.close()

def plotzero():
    zero_count = (df == 0).sum()
    plt.figure(figsize=(12, 6))
    zero_count.plot(kind='bar', color='skyblue')
    plt.title('Zero Count (\'0\')')
    plt.ylabel('Count of Zero Values')
    plt.xlabel('Column Name')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"./metricsplot/zero_values_.png")
    plt.close()
    
    print(f"Number of rows in dataframe: {len(df)}")

def get_feature_names():
    try:
        vectorizer = joblib.load("./nutrition_models/tfidf_vectorizer.joblib")
        tfidf_features = vectorizer.get_feature_names_out().tolist()
        try:
            with open("./nutrition_models/metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            numeric_features = metadata.get('input_features', ['protein', 'total_fat', 'carbohydrate', 'sodium', 'cholesterol'])
        except:
            print("Warning: Could not load metadata, using default numeric features")
            numeric_features = ['protein', 'total_fat', 'carbohydrate', 'sodium', 'cholesterol']
        all_features = tfidf_features + numeric_features
        
        print(f"Found {len(tfidf_features)} TF-IDF features and {len(numeric_features)} numeric features")
        print(f"Total features: {len(all_features)}")
        
        return all_features
        
    except Exception as e:
        print(f"Error loading feature names: {e}")
        return None

def debug_feature_mapping(model):
    feature_names = get_feature_names()
    
    if feature_names:
        print(f"Total feature names available: {len(feature_names)}")
        print("First 10 feature names:", feature_names[:10])
        print("Last 10 feature names:", feature_names[-10:])
    first_estimator = model.estimators_[0]
    booster = first_estimator.get_booster()
    importance_dict = booster.get_score(importance_type='weight')
    
    print(f"XGBoost reports {len(importance_dict)} features")
    print("Sample XGBoost feature names:", list(importance_dict.keys())[:10])
    if feature_names:
        max_xgb_idx = max([int(name[1:]) for name in importance_dict.keys() if name.startswith('f')])
        print(f"Highest XGBoost feature index: {max_xgb_idx}")
        print(f"Available feature names: {len(feature_names)}")

def plot_combined_feature_importance(model, save_path):
    feature_names = get_feature_names()
    
    n_outputs = len(model.estimators_)
    n_cols = 4
    n_rows = math.ceil(n_outputs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_outputs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, estimator in enumerate(model.estimators_):
        booster = estimator.get_booster()
        importance_dict = booster.get_score(importance_type='weight')
        
        if feature_names and len(feature_names) > 0:
            mapped_importance = {}
            unmapped_count = 0
            
            for generic_name, score in importance_dict.items():
                try:
                    if generic_name.startswith('f'):
                        feature_idx = int(generic_name[1:])  
                        if feature_idx < len(feature_names):
                            meaningful_name = feature_names[feature_idx]
                            if len(meaningful_name) > 20:
                                meaningful_name = meaningful_name[:17] + "..."
                            mapped_importance[meaningful_name] = score
                        else:
                            mapped_importance[generic_name] = score
                            unmapped_count += 1
                    else:
                        mapped_importance[generic_name] = score
                except Exception as e:
                    print(f"Error mapping {generic_name}: {e}")
                    mapped_importance[generic_name] = score
                    unmapped_count += 1
            
            if unmapped_count > 0:
                print(f"Warning: {unmapped_count} features could not be mapped for output {i}")
            sorted_features = sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            
            if sorted_features:
                names, scores = zip(*sorted_features)
                axes[i].barh(range(len(names)), scores, color='salmon')
                axes[i].set_yticks(range(len(names)))
                axes[i].set_yticklabels(names, fontsize=8)
                axes[i].set_xlabel('Importance (Weight)')
                axes[i].set_title(f"Output #{i}", fontsize=10)
                axes[i].grid(axis='x', alpha=0.3)
                axes[i].invert_yaxis()
            else:
                axes[i].text(0.5, 0.5, 'No features found', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"Output #{i}", fontsize=10)
            
        else:
            try:
                xgb.plot_importance(
                    booster,
                    importance_type='weight',
                    max_num_features=20,
                    title=f"Output #{i}",
                    ax=axes[i]
                )
                axes[i].title.set_size(10)
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Plot error: {str(e)}', ha='center', va='center', transform=axes[i].transAxes)
    for j in range(n_outputs, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to: {save_path}")
model = joblib.load("./nutrition_models/model1_text_numeric.joblib")

plot_combined_feature_importance(model, "./metricsplot/feature_importance_model_1.png")

