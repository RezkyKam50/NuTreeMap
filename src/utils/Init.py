from dash import (
    Dash, 
    html, 
    dcc, 
    Input, 
    Output, 
    State, 
    callback, 
    MATCH, 
    ALL
)
from utils.NuXG.xgb_infer import (
    load_models_and_components, 
    predict_with_loaded_models
)
from utils.Constellation.Loader import LoadDataset


def Init():
    plot_df_3d, plot_df_2d = LoadDataset()
    loaded_models = load_models_and_components()
    all_names = sorted(plot_df_3d['name'].dropna().unique())
    all_food_types_1 = sorted(plot_df_3d['food_type_1'].dropna().unique())
    all_food_types_2 = sorted(plot_df_3d['food_type_2'].dropna().unique())

    return plot_df_3d, plot_df_2d, loaded_models, all_names, all_food_types_1, all_food_types_2

