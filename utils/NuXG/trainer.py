import cudf
import cupy as cp
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.model_selection import train_test_split
from cuml.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer
from sklearn.multioutput import MultiOutputRegressor
from scipy.sparse import hstack
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import joblib
import os
import datetime

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.input_features = ['protein', 'total_fat', 'carbohydrate', 'sodium', 'cholesterol']
        self.target_features = ['calories', 'calcium', 'saturated_fat', 'fiber', 'saturated_fatty_acids', 
                               'monounsaturated_fatty_acids', 'polyunsaturated_fatty_acids', 'fatty_acids_total_trans']
        self.all_nutrients = ['protein', 'calories', 'calcium', 'total_fat', 'saturated_fat', 'cholesterol', 
                             'sodium', 'carbohydrate', 'fiber', 'saturated_fatty_acids', 
                             'monounsaturated_fatty_acids', 'polyunsaturated_fatty_acids', 
                             'fatty_acids_total_trans']
    
    def load_data(self, filepath):
        df = cudf.read_parquet(filepath)
        return df[['protein', 'calories', 'calcium', 'total_fat', 'saturated_fat', 'cholesterol', 'sodium',
                  'carbohydrate', 'fat', 'fiber', 'saturated_fatty_acids', 'monounsaturated_fatty_acids',
                  'polyunsaturated_fatty_acids', 'fatty_acids_total_trans', 'name', 'food_type_1', 'food_type_2']]
    
    def create_text_features(self, df):
        df['combined_text'] = df['name'].astype(str) + ', ' + df['food_type_1'].fillna('').astype(str) + ', ' + df['food_type_2'].fillna('').astype(str)
        return self.vectorizer.fit_transform(df['combined_text'])
    
    def prepare_model_data(self, df, food_name_features):
        X_numeric = df[self.input_features].values
        X_combined = hstack([food_name_features, X_numeric])
        y_subset = df[self.target_features].values
        
        X_text_only = food_name_features
        y_all = df[self.all_nutrients].values
        
        return X_combined, y_subset, X_text_only, y_all

class ModelTrainer:
    def __init__(self):
        self.base_model = XGBRegressor(
            objective='reg:squarederror', 
            random_state=42, 
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
    
    def create_models(self):
        return MultiOutputRegressor(self.base_model), MultiOutputRegressor(self.base_model)
    
    def train_models(self, model1, model2, X_train_1, y_train_1, X_train_2, y_train_2):
        model1.fit(X_train_1, y_train_1)
        model2.fit(X_train_2, y_train_2)
        return model1, model2
    
    def evaluate_model(self, model, X_test, y_test, target_names):
        y_pred = model.predict(X_test)
        metrics = {}
        for i, target in enumerate(target_names):
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            metrics[target] = {'MAE': mae, 'MSE': mse, 'R2': r2}
            print(f"{target:25} | MAE: {mae:7.2f} | MSE: {mse:10.2f} | R²: {r2:6.3f}")
        return metrics, y_pred

class CrossValidator:
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds
        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    def multioutput_r2_scorer(self, y_true, y_pred):
        scores = []
        for i in range(y_true.shape[1]):
            scores.append(r2_score(y_true[:, i], y_pred[:, i]))
        return cp.mean(scores)
    
    def multioutput_mae_scorer(self, y_true, y_pred):
        scores = []
        for i in range(y_true.shape[1]):
            scores.append(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        return cp.mean(scores)
    
    def cross_validate(self, model, X, y):
        r2_scorer = make_scorer(self.multioutput_r2_scorer, greater_is_better=True)
        mae_scorer = make_scorer(self.multioutput_mae_scorer, greater_is_better=False)
        
        cv_r2_scores = cross_val_score(model, X, y, cv=self.kfold, scoring=r2_scorer, n_jobs=-1)
        cv_mae_scores = cross_val_score(model, X, y, cv=self.kfold, scoring=mae_scorer, n_jobs=-1)
        
        return cv_r2_scores, cv_mae_scores

class ModelPredictor:
    def __init__(self, model1, model2, vectorizer, input_features, target_features, all_nutrients):
        self.model1 = model1
        self.model2 = model2
        self.vectorizer = vectorizer
        self.input_features = input_features
        self.target_features = target_features
        self.all_nutrients = all_nutrients
    
    def predict_new_food(self, food_text, numeric_features):
        new_food_features = self.vectorizer.transform([food_text])
        
        new_numeric = cp.array([numeric_features])
        new_X_combined = hstack([new_food_features, new_numeric])
        new_X_combined_dense = new_X_combined.toarray()
        
        predicted_values_1 = self.model1.predict(new_X_combined_dense)[0]
        predicted_values_2 = self.model2.predict(new_food_features.toarray())[0]
        
        return predicted_values_1, predicted_values_2
    
    def print_predictions(self, food_text, predicted_values_1, predicted_values_2):
        print(f"\nModel 1 Predictions for '{food_text}':")
        print("-" * 50)
        for i, target in enumerate(self.target_features):
            print(f"{target:25}: {predicted_values_1[i]:8.2f}")
        
        print(f"\nModel 2 Predictions for '{food_text}':")
        print("-" * 50)
        for i, nutrient in enumerate(self.all_nutrients):
            print(f"{nutrient:25}: {predicted_values_2[i]:8.2f}")

class ModelSaver:
    @staticmethod
    def save_models_and_components(model1, model2, vectorizer, target_features, all_nutrients, input_features, save_dir="nutrition_models"):
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
                'model1_type': 'MultiOutputRegressor with XGBoost',
                'model2_type': 'MultiOutputRegressor with XGBoost',
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

class Visualizer:
    @staticmethod
    def plot_cross_validation_metrics(cv_r2_scores_1, cv_mae_scores_1, cv_r2_scores_2, cv_mae_scores_2, cv_folds=5):
        cv_df = cudf.DataFrame({
            "Fold": list(range(1, cv_folds + 1)) * 2,
            "Model": ["Model 1"] * cv_folds + ["Model 2"] * cv_folds,
            "R2": cp.concatenate([cv_r2_scores_1, cv_r2_scores_2]),
            "MAE": -cp.concatenate([cv_mae_scores_1, cv_mae_scores_2])   
        })
        
        cv_summary = cv_df.groupby("Model").agg({
            "R2": ['mean', 'std'],
            "MAE": ['mean', 'std']
        }).reset_index()
        
        cv_summary.columns = ['Model', 'R2_mean', 'R2_std', 'MAE_mean', 'MAE_std']
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        x = cp.arange(len(cv_summary))
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
        plt.savefig("./docs/reports/metricsplot/metrics_dual_axis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_individual_target_performance(metrics1, target_features, metrics2, all_nutrients):
        df_model1 = cudf.DataFrame({
            "Target": target_features,
            "MAE": [metrics1[target]['MAE'] for target in target_features],
            "R2": [metrics1[target]['R2'] for target in target_features]
        })
        
        df_model2 = cudf.DataFrame({
            "Target": all_nutrients,
            "MAE": [metrics2[nutrient]['MAE'] for nutrient in all_nutrients],
            "R2": [metrics2[nutrient]['R2'] for nutrient in all_nutrients]
        })
        
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
        plt.savefig("./docs/reports/metricsplot/individual_target_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

class NutritionPredictor:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.cross_validator = CrossValidator()
        self.saver = ModelSaver()
        self.visualizer = Visualizer()
    
    def run_pipeline(self):
        print("Loading dataset...")
        df = self.preprocessor.load_data("umapped3D.parquet")
        
        print("Creating text features...")
        food_name_features = self.preprocessor.create_text_features(df)
        
        print("Preparing model data...")
        X_combined, y_subset, X_text_only, y_all = self.preprocessor.prepare_model_data(df, food_name_features)
        
        X_combined_dense = X_combined.toarray()
        X_text_dense = X_text_only.toarray()
        
        print("Performing cross-validation...")
        model1, model2 = self.trainer.create_models()
        
        cv_r2_scores_1, cv_mae_scores_1 = self.cross_validator.cross_validate(model1, X_combined_dense, y_subset)
        cv_r2_scores_2, cv_mae_scores_2 = self.cross_validator.cross_validate(model2, X_text_dense, y_all)
        
        print("Splitting data for training and testing...")
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_combined_dense, y_subset, test_size=0.2, random_state=42)
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_text_dense, y_all, test_size=0.2, random_state=42)
        
        print("Training models...")
        model1, model2 = self.trainer.train_models(model1, model2, X_train_1, y_train_1, X_train_2, y_train_2)
        
        print("\nModel 1 Evaluation:")
        metrics1, y_pred_1 = self.trainer.evaluate_model(model1, X_test_1, y_test_1, self.preprocessor.target_features)
        
        print("\nModel 2 Evaluation:")
        metrics2, y_pred_2 = self.trainer.evaluate_model(model2, X_test_2, y_test_2, self.preprocessor.all_nutrients)
        
        print("\nMaking predictions on new food...")
        predictor = ModelPredictor(model1, model2, self.preprocessor.vectorizer, 
                                 self.preprocessor.input_features, self.preprocessor.target_features, 
                                 self.preprocessor.all_nutrients)
        
        new_food_numeric = [25.0, 3.0, 120.0, 20.0, 100.0]
        pred1, pred2 = predictor.predict_new_food("Chicken Salad, Salted", new_food_numeric)
        predictor.print_predictions("Chicken Salad, Salted", pred1, pred2)
        
        print("\nSaving models...")
        self.saver.save_models_and_components(model1, model2, self.preprocessor.vectorizer,
                                            self.preprocessor.target_features, self.preprocessor.all_nutrients,
                                            self.preprocessor.input_features)
        
        print("\nGenerating visualizations...")
        self.visualizer.plot_cross_validation_metrics(cv_r2_scores_1, cv_mae_scores_1, cv_r2_scores_2, cv_mae_scores_2)
        self.visualizer.plot_individual_target_performance(metrics1, self.preprocessor.target_features, 
                                                         metrics2, self.preprocessor.all_nutrients)
        
        print("Pipeline completed successfully!")

if __name__ == "__main__":
    os.makedirs("./docs/reports/metricsplot", exist_ok=True)
    pipeline = NutritionPredictor()
    pipeline.run_pipeline()