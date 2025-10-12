from dash import Input, Output, State, callback, html
from utils.NuXG.xgb_infer import predict_with_loaded_models
import dash_bootstrap_components as dbc
import pandas as pd
from typing import Optional


@callback(
    Output('prediction-output', 'children'),
    Output('prediction-output', 'key'),  # Add key output to force re-render
    Input('predict-btn', 'n_clicks'),
    [State('food-name-input', 'value'),
     State('protein-input', 'value'),
     State('fat-input', 'value'),
     State('carbs-input', 'value'),
     State('sodium-input', 'value'),
     State('cholesterol-input', 'value')],
    prevent_initial_call=True
)
def predict_nutrients(n_clicks: int, food_name: str, protein: float, fat: float,
                     carbs: float, sodium: float, cholesterol: float):
    from app import loaded_models
    try:
        numeric_values = {
            "protein": protein,
            "total_fat": fat,
            "carbohydrate": carbs,
            "sodium": sodium,
            "cholesterol": cholesterol
        }
        
        if all(v == 0.0 for v in numeric_values.values()):
            numeric_values = None
        
        results = predict_with_loaded_models(
            food_name=food_name,
            numeric_values=numeric_values,
            loaded_components=loaded_models
        )
        
        model_type = "Text-Only Model (2 nutrients)" if numeric_values is None else "Mixed Model (13 nutrients)"
        
        output = [
            dbc.Alert(f"Used {model_type} for: {results['food_name']}", color="success")
        ]
        
        if results['model1_predictions']:
            model1_data = []
            for k, v in results['model1_predictions'].items():
                if k != 'calcium':
                    unit = 'kJ' if k == 'calories' else 'g'
                    model1_data.append({
                        'Nutrient': k.replace('_', ' ').title(),
                        'Value': f"{max(v, 0):.2f}",
                        'Unit': unit
                    })
            
            output.extend([
                html.H6("XGBoost: Text (TF-IDF + Numerical)", className="mt-3"),
                html.Small("7 nutrients predicted", className="text-muted"),
                dbc.Table.from_dataframe(pd.DataFrame(model1_data), 
                                        striped=True, bordered=True, hover=True, size='sm')
            ])
        
        if results['model2_predictions'] and numeric_values is None:
            model2_data = []
            for k, v in results['model2_predictions'].items():
                if k in ['protein', 'carbohydrate']:
                    model2_data.append({
                        'Nutrient': k.replace('_', ' ').title(),
                        'Value': f"{max(v, 0):.2f}",
                        'Unit': 'g'
                    })
            
            output.extend([
                html.H6("XGBoost: Text (TF-IDF)", className="mt-3"),
                html.Small("2 nutrients predicted", className="text-muted"),
                dbc.Table.from_dataframe(pd.DataFrame(model2_data), 
                                        striped=True, bordered=True, hover=True, size='sm')
            ])
        
        # Return both the content and a unique key based on n_clicks to force re-render
        return html.Div(output, className="prediction-animate"), str(n_clicks)
        
    except Exception as e:
        return dbc.Alert(f"❌ Error: {str(e)}", color="danger"), str(n_clicks)