import joblib, os, pickle, numpy as np
from scipy.sparse import hstack

def load_models_and_components(save_dir="nutrition_models"):
    model1 = joblib.load(os.path.join(save_dir, "model1_text_numeric.joblib"))
    model2 = joblib.load(os.path.join(save_dir, "model2_text_only.joblib"))
    vectorizer = joblib.load(os.path.join(save_dir, "tfidf_vectorizer.joblib"))
    # imputer_subset = joblib.load(os.path.join(save_dir, "imputer_subset.joblib"))
    # imputer_all = joblib.load(os.path.join(save_dir, "imputer_all.joblib"))

    with open(os.path.join(save_dir, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)
    
    print("✓ All components loaded successfully!")
    print(f"Models saved on: {metadata['save_date']}")
    
    return {
        'model1': model1,
        'model2': model2,
        'vectorizer': vectorizer,
        # 'imputer_subset': imputer_subset,
        # 'imputer_all': imputer_all,
        'metadata': metadata
    }


def predict_with_loaded_models(food_name, numeric_values=None, loaded_components=None, save_dir="nutrition_models"):
    if loaded_components is None:
        loaded_components = load_models_and_components(save_dir)
    
    model1 = loaded_components['model1']
    model2 = loaded_components['model2']
    vectorizer = loaded_components['vectorizer']
    metadata = loaded_components['metadata']
    food_text_features = vectorizer.transform([food_name])
    predictions_model2 = model2.predict(food_text_features.toarray())[0]
    model2_predictions = {}
    for i, nutrient in enumerate(metadata['all_nutrients']):
        model2_predictions[nutrient] = predictions_model2[i]
    results = {
        'food_name': food_name,
        'model2_predictions': model2_predictions,
        'model1_predictions': None,
        'input_features_used': None,
        'metadata': metadata
    }
    if numeric_values is not None:
        input_features = metadata['input_features']
        numeric_array = np.array([[numeric_values.get(feature, 0) for feature in input_features]])
        combined_features = hstack([food_text_features, numeric_array])
        combined_features_dense = combined_features.toarray()
        predictions_model1 = model1.predict(combined_features_dense)[0]
        model1_predictions = {}
        for i, nutrient in enumerate(metadata['target_features']):
            model1_predictions[nutrient] = predictions_model1[i]
        input_features_used = {}
        for feature in input_features:
            value = numeric_values.get(feature, 0)
            input_features_used[feature] = value
        results['model1_predictions'] = model1_predictions
        results['input_features_used'] = input_features_used
    else:
        for feature in metadata['input_features']:
            print(f"  - {feature}")
    return results


if __name__ == "__main__":
    loaded_components = load_models_and_components()
    results = predict_with_loaded_models(
        food_name="Chicken Salad Salted",
        numeric_values={
            'protein': 25.0,
            'total_fat': 3.0,
            'carbohydrate': 120.0,
            'sodium': 20.0,
            'cholesterol': 100.0
        },
        loaded_components=loaded_components
    )

    print(f"\nReturned results structure:")
    print(f"Food name: {results['food_name']}")
    print(f"Model 2 predictions available: {len(results['model2_predictions'])} nutrients")
    if results['model1_predictions']:
        print(f"Model 1 predictions available: {len(results['model1_predictions'])} nutrients")
    else:
        print("Model 1 predictions: Not available")