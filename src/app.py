from dash import (
    Dash, 
    html, 
    dcc, 
    Input, 
    Output, 
    State, 
    callback, 
    MATCH, 
    ALL,
    clientside_callback,
    ctx
)
from utils.config.config import (
    PLOT_HEIGHT, 
    NEAREST_NEIGHBORS, 
    MAX_SUGGESTIONS
)
from utils.Init import Init
import dash_bootstrap_components as dbc

from datetime import datetime, timedelta

plot_df_3d, plot_df_2d, loaded_models, all_names, all_food_types_1, all_food_types_2 = Init()
 
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], assets_folder='assets')


app.title = "NuTreeMap"
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(
                    src="/assets/Logo.png",
                    style={
                        "height": "100px",
                        "marginRight": "15px"
                    }
                ),
                html.H1(
                "The Nutritionist Analytics Dashboard.",
                className="company-text"
                )
            ], className="app-title d-flex align-items-center justify-content-center my-4")
        ])
    ]),
    
    # Separator and NuMAP Section Title
    html.Hr(style={'borderColor': '#00ffaf', 'marginTop': '20px', 'marginBottom': '20px'}),
    dbc.Row([
        dbc.Col([
            html.H2("Constellation", style={'color': '#00ffaf', 'marginBottom': '20px'})
        ])
    ]),
    
    # NuMAP Section 
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
                    
                    # Dynamic Search Sections
                    html.Div(id='search-sections-container', children=[
                        # Initial search section
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H5("Food 1", style={'color': '#00ffaf'}),
                                    dbc.Input(
                                        id={'type': 'search-input', 'index': 0}, 
                                        placeholder="Search food...", 
                                        value="", 
                                        className="mb-2",
                                        style={
                                            'backgroundColor': '#3A4249',
                                            'color': '#00ffaf',
                                            'borderColor': '#00ffaf',
                                            'borderWidth': '1px'
                                        }
                                    ),
                                    dcc.Dropdown(
                                        id={'type': 'search-dropdown', 'index': 0}, 
                                        placeholder="Select from results...",
                                        style={
                                            'backgroundColor': '#3A4249',
                                            'color': '#00ffaf',
                                            'borderColor': '#00ffaf',
                                            'borderWidth': '1px'
                                        },
                                        className="custom-dropdown",
                                    )
                                ], className='glow-card search-section')
                            ], width=6)
                        ], className="mb-3")
                    ]),
                    
                    # Add/Remove Food Buttons
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Add Food", id="add-food-btn", color="success", size="sm", className="me-2"),
                            dbc.Button("Remove Food", id="remove-food-btn", color="danger", size="sm")
                        ], width=12, className="mb-3")
                    ], id="food-controls", style={'display': 'none'}),
                    
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
                    
                    # Plot and Bar Chart Side by Side
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                id='main-plot', 
                                style={'height': f'{PLOT_HEIGHT}px'},
                                config={'scrollZoom': True, 'displayModeBar': True},
                                clear_on_unhover=True
                            )
                        ], width=8),
                        dbc.Col([
                            dcc.Graph(
                                id='cluster-barplot',
                                className="cluster-barplot",
                                config={'displayModeBar': True},
                                style={
                                    'overflowX': 'auto',   # enable vertical scroll
                                    'overflowY': 'auto',   # enable vertical scroll
                                }
                            )
                        ], width=4)
                    ]),
                    # Results Table
                    html.Div(id='results-table', className="results-container mt-4", style={'backgroundColor': 'transparent'})
                ])
            ], style={'backgroundColor': '#2A3238', 'border': '1px solid #00ffaf'})
        ])
    ], className="mb-4"),

 
    html.Hr(style={'borderColor': '#00ffaf', 'marginTop': '40px', 'marginBottom': '20px'}),
    
    # Globe Section 
    dbc.Row([
        dbc.Col([
            html.H2("GAIA-GX", style={'color': '#00ffaf', 'marginBottom': '20px'})
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
 
                    html.Div(className="cosmic-dust"),
                    html.Div(className="galaxy-bg"),
                    
                    # Hidden data stores for progressive loading
                    dcc.Store(id='news-data-store'),
                    dcc.Store(id='weather-data-store'),
                    dcc.Store(id='earthquake-data-store'),
                    dcc.Store(id='tide-data-store'),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Region:", style={'color': 'white'}),
                            dcc.Dropdown(
                                id='region-selector',
                                options=[
                                    {'label': 'North America', 'value': 'NA'},
                                    {'label': 'South America', 'value': 'SA'},
                                    {'label': 'Europe', 'value': 'EU'},
                                    {'label': 'Africa', 'value': 'AF'},
                                    {'label': 'Asia', 'value': 'AS'},
                                    {'label': 'Oceania', 'value': 'OC'}
                                ],
                                value='NA',
                                style={
                                    'backgroundColor': '#3A4249',
                                    'color': '#00ffaf',
                                    'borderColor': '#00ffaf'
                                },
                                className="mb-3"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Visualization Mode:", style={'color': 'white'}),
                            dcc.Dropdown(
                                id='globe-mode',
                                options=[
                                    {'label': 'Geographic', 'value': 'geo'},
                                    {'label': 'Orthographic', 'value': 'ortho'},
                                    {'label': 'Natural Earth', 'value': 'natural'}
                                ],
                                value='ortho',
                                style={
                                    'backgroundColor': '#3A4249',
                                    'color': '#00ffaf',
                                    'borderColor': '#00ffaf'
                                },
                                className="mb-3"
                            )
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                id='earth-globe',
                                style={'height': '1200px'},
                                config={'displayModeBar': True, 'scrollZoom': True}
                            )
                        ])
                    ])
                ], className="globe-container")
            ], className='globe-card', style={'backgroundColor': '#2A3238', 'border': '2px solid #00ffaf', 'position': 'relative'})
        ])
    ], className="mb-4"),

    html.Hr(style={'borderColor': '#00ffaf', 'marginTop': '40px', 'marginBottom': '20px'}),
 
    dbc.Row([
        dbc.Col([
            html.H2("Nu-XG", style={'color': '#00ffaf', 'marginBottom': '20px'}, className="nutrixg-title")
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
                            html.Br(),

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
                            
                            dbc.Button("Predict Nutrients", id="predict-btn", 
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
            ], className='nutrixg-card')
        ])
    ], className="mb-4")
], fluid=True)

 
from utils.Constellation.SearchBlock import update_dropdown, update_search_sections, create_search_section
from utils.Constellation.Clustering import update_plot
from utils.NuXG.XGBRegression import predict_nutrients
import utils.GAIAGX.Globe 
 
@callback(
    Output({'type': 'search-dropdown', 'index': MATCH}, 'options'),
    Input({'type': 'search-input', 'index': MATCH}, 'value'),
    prevent_initial_call=True
)
def update_dropdown_callback(search_value):
    return update_dropdown(search_value)

if __name__ == '__main__':
    app.run(debug=True, port=8080)