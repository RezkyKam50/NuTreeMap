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
from utils.config.config import (
    PLOT_HEIGHT, 
    NEAREST_NEIGHBORS, 
    MAX_SUGGESTIONS
)
from utils.Init import Init
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Initialize data
plot_df_3d, plot_df_2d, loaded_models, all_names, all_food_types_1, all_food_types_2 = Init()

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "NuTreeMap"

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("NuTreeMap", className="text-center my-4")
        ])
    ]),
    
    # Separator and NuMAP Section Title
    html.Hr(style={'borderColor': '#00ffaf', 'marginTop': '20px', 'marginBottom': '20px'}),
    dbc.Row([
        dbc.Col([
            html.H2("NuMAP", style={'color': '#00ffaf', 'marginBottom': '20px'})
        ])
    ]),
    
    # NuMAP Section - Combined Search and Visualization
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    # Visualization Controls
                    dbc.Row([
                        dbc.Col([
                            dbc.RadioItems(
                                id='plot-type',
                                options=[
                                    {'label': '3D UMAP', 'value': '3d'},
                                    {'label': '2D UMAP', 'value': '2d'}
                                ],
                                value='3d',
                                inline=True
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Checklist(
                                id='comparison-mode',
                                options=[{'label': 'Enable food comparison mode', 'value': 'compare'}],
                                value=[]
                            )
                        ], width=6)
                    ], className="mb-3"),
                    
                    # Search Section - Always render both, hide with CSS
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("Food 1", style={'color': '#00ffaf'}),
                                dbc.Input(
                                    id='search-1', 
                                    placeholder="Search food...", 
                                    value="Beef", 
                                    className="mb-2",
                                    style={
                                        'backgroundColor': '#3A4249',
                                        'color': '#00ffaf',
                                        'borderColor': '#00ffaf',
                                        'borderWidth': '1px'
                                    }
                                ),
                                dcc.Dropdown(
                                    id='dropdown-1', 
                                    placeholder="Select from results...",
                                    style={
                                        'backgroundColor': '#3A4249',
                                        'color': '#00ffaf',
                                        'borderColor': '#00ffaf',
                                        'borderWidth': '1px'
                                    },
                                    className="custom-dropdown",
                                )
                            ], id='search-section-1', style={'backgroundColor': '#3A4249', 'padding': '15px', 'borderRadius': '5px'})
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.H5("Food 2", style={'color': '#00ffaf'}),
                                dbc.Input(
                                    id='search-2', 
                                    placeholder="Search food...", 
                                    value="Chicken", 
                                    className="mb-2",
                                    style={
                                        'backgroundColor': '#3A4249',
                                        'color': '#00ffaf',
                                        'borderColor': '#00ffaf',
                                        'borderWidth': '1px'
                                    }
                                ),
                                dcc.Dropdown(
                                    id='dropdown-2', 
                                    placeholder="Select from results...",
                                    style={
                                        'backgroundColor': '#3A4249',
                                        'color': '#00ffaf',
                                        'borderColor': '#00ffaf',
                                        'borderWidth': '1px'
                                    },
                                    className="custom-dropdown",
                                )
                            ], id='search-section-2', style={
                                'display': 'none', 
                                'backgroundColor': '#3A4249', 
                                'padding': '15px', 
                                'borderRadius': '5px'
                            })
                        ], width=6)
                    ], className="mb-3"),
                    
                    # Control Options
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id='show-cluster-only',
                                options=[{'label': 'Show cluster only', 'value': 'cluster_only'}],
                                value=[]  # Default: show all clusters
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Checklist(
                                id='show-outliers',
                                options=[{'label': 'Show outliers (cluster = -1)', 'value': 'outliers'}],
                                value=[]  # Default: hide outliers
                            )
                        ], width=6)
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Submit", id="submit-btn", color="primary", className="mb-3")
                        ])
                    ]),
                    
                    html.P("Clustering is done purely by nutritional content, not by food name or type", 
                           className="text-muted small"),
                    
                    # Plot
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='main-plot', style={'height': f'{PLOT_HEIGHT}px'})
                        ])
                    ]),
                    
                    # Results Table
                    html.Div(id='results-table', className="results-container mt-4", style={'backgroundColor': 'transparent'})
                ])
            ], style={'backgroundColor': '#2A3238', 'border': '1px solid #00ffaf'})
        ])
    ], className="mb-4"),

    html.Hr(),

    # Nutri-XG Predictor - Always Visible
    dbc.Row([
        dbc.Col([
            html.H3("Nutri-XG Predictor", style={'color': '#00ffaf', 'marginBottom': '20px'})
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Input Food Information", style={'color': '#00ffaf'}),
                            dbc.Input(
                                id='food-name-input',
                                placeholder="e.g., Potato, Salted",
                                value="Potato, Salted",
                                className="mb-2",
                                style={
                                    'backgroundColor': '#3A4249',
                                    'color': '#00ffaf',
                                    'borderColor': '#00ffaf',
                                    'borderWidth': '1px'
                                }
                            ),
                            html.Small("Syntax: [Meat/Liquids][Preparation][Seasoning]", 
                                    className="text-muted"),
                            
                            html.H5("Known Nutritional Values", className="mt-4", style={'color': '#00ffaf'}),
                            html.Small("Enter known values. Leave as 0 if unknown.", 
                                    className="text-muted mb-3"),
                            
                            dbc.Label("Protein (g)", style={'color': 'white'}),
                            dbc.Input(
                                id='protein-input', 
                                type='number', 
                                value=25.0, 
                                step=0.1, 
                                min=0,
                                style={
                                    'backgroundColor': '#3A4249',
                                    'color': '#00ffaf',
                                    'borderColor': '#00ffaf',
                                    'borderWidth': '1px'
                                }
                            ),
                            
                            dbc.Label("Total Fat (g)", className="mt-2", style={'color': 'white'}),
                            dbc.Input(
                                id='fat-input', 
                                type='number', 
                                value=3.0, 
                                step=0.1, 
                                min=0,
                                style={
                                    'backgroundColor': '#3A4249',
                                    'color': '#00ffaf',
                                    'borderColor': '#00ffaf',
                                    'borderWidth': '1px'
                                }
                            ),
                            
                            dbc.Label("Carbohydrates (g)", className="mt-2", style={'color': 'white'}),
                            dbc.Input(
                                id='carbs-input', 
                                type='number', 
                                value=120.0, 
                                step=0.1, 
                                min=0,
                                style={
                                    'backgroundColor': '#3A4249',
                                    'color': '#00ffaf',
                                    'borderColor': '#00ffaf',
                                    'borderWidth': '1px'
                                }
                            ),
                            
                            dbc.Label("Sodium (g)", className="mt-2", style={'color': 'white'}),
                            dbc.Input(
                                id='sodium-input', 
                                type='number', 
                                value=20.0, 
                                step=0.1, 
                                min=0,
                                style={
                                    'backgroundColor': '#3A4249',
                                    'color': '#00ffaf',
                                    'borderColor': '#00ffaf',
                                    'borderWidth': '1px'
                                }
                            ),
                            
                            dbc.Label("Cholesterol (g)", className="mt-2", style={'color': 'white'}),
                            dbc.Input(
                                id='cholesterol-input', 
                                type='number', 
                                value=10.0, 
                                step=0.1, 
                                min=0,
                                style={
                                    'backgroundColor': '#3A4249',
                                    'color': '#00ffaf',
                                    'borderColor': '#00ffaf',
                                    'borderWidth': '1px'
                                }
                            ),
                            
                            dbc.Button("🔮 Predict Nutrients", id="predict-btn", 
                                    color="primary", className="mt-3 w-100")
                        ], width=6),
                        
                        dbc.Col([
                            html.H5("Prediction Results", style={'color': '#00ffaf'}),
                            html.Div(
                                id='prediction-output',
                                style={
                                    'backgroundColor': '#3A4249',
                                    'padding': '15px',
                                    'borderRadius': '5px',
                                    'border': '1px solid #00ffaf',
                                    'color': 'white',
                                    'minHeight': '300px'
                                }
                            )
                        ], width=6)
                    ])
                ])
            ], style={'backgroundColor': '#2A3238', 'border': '1px solid #00ffaf'})
        ])
    ], className="mt-4")
], fluid=True, style={'backgroundColor': "#3A4249", 'color': 'white', 'minHeight': '100vh'})

# Import callbacks after app is defined
# This is important to avoid circular imports
from utils.SearchBlock import (
    toggle_search_sections,
    update_dropdown_1, 
    update_dropdown_2
)
from utils.Clustering import update_plot
from utils.XGBRegression import predict_nutrients

if __name__ == '__main__':
    app.run(debug=True, port=8080)