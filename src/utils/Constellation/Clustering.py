from dash import Input, Output, State, callback, MATCH, ALL
from utils.config.config import PLOT_HEIGHT, NEAREST_NEIGHBORS
from utils.Cmap import get_color_mapping
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dash import html
 
COLORS = {
    'background': {
        'plot': 'rgb(42, 50, 56)', # for 2d plot bg
        'paper': 'rgba(0, 0, 0, 0)',
        'scene': 'rgba(0, 0, 0, 0)', # for 3d plot bg
        'axis': 'rgba(0, 0, 0, 0)',
        'legend': 'rgb(58, 66, 73)'
    },
    'primary': {
        'main': '#00ffaf',
        'border': "#FAFFFB",
        'grid': 'rgba(0,255,65,0.1)',
        'grid_3d': 'rgba(0, 0, 0, 0)',
        'zeroline': 'rgba(0,255,65,0.5)',
        'highlight': 'rgba(0, 255, 65, 0.0)',
        'line': 'rgba(0, 255, 65, 0.5)'
    },
    'marker': {
        'line': 'rgba(0, 255, 175, 0.00)'
    },
    'comparison': {
        'colors': ['#4A90E2', '#50E3C2', '#B8E986', '#F5A623', '#D0021B', '#9013FE', '#F8E71C', '#7ED321'],
        'borders': ['#2C3E50', '#2C3E50', '#2C3E50', '#2C3E50', '#2C3E50', '#2C3E50', '#2C3E50', '#2C3E50']
    }
}

def create_cluster_barplot(df: pd.DataFrame, color_mapping: Dict[str, str]) -> go.Figure:
 
    cluster_counts = df['cluster'].value_counts().sort_values(ascending=True)
 
    clusters = cluster_counts.index.astype(str).tolist()
    counts = cluster_counts.values.tolist()
    colors = [color_mapping.get(str(c), COLORS['primary']['main']) for c in cluster_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=counts,
            y=clusters,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(width=1, color=COLORS['primary']['border'])
            ),
            text=counts,
            textposition='outside',
            textfont=dict(color=COLORS['primary']['main']),
            hovertemplate='Cluster %{x}<br>Count: %{y}<extra></extra>'
        )
    ])

    
    
    fig.update_layout(
        title=dict(
            text='Cluster Distribution',
            x=0.5,
            xanchor='center',
            font=dict(size=16, color=COLORS['primary']['main'])
        ),
        xaxis=dict(
            title='Cluster',
            showline=True,
            linewidth=2,
            linecolor=COLORS['primary']['main'],
            mirror=True,
            gridcolor=COLORS['primary']['grid'],
            tickfont=dict(color=COLORS['primary']['main']),
        ),
        yaxis=dict(
            title='Count',
            showline=True,
            linewidth=2,
            linecolor=COLORS['primary']['main'],
            mirror=True,
            gridcolor=COLORS['primary']['grid'],
            tickfont=dict(color=COLORS['primary']['main']),
        ),
        plot_bgcolor=COLORS['background']['plot'],
        paper_bgcolor=COLORS['background']['paper'],
        font=dict(color=COLORS['primary']['main'], family="Arial, sans-serif"),
        showlegend=False,
        height=PLOT_HEIGHT,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

 
@callback(
    [Output('main-plot', 'figure'),
     Output('cluster-barplot', 'figure'),
     Output('results-table', 'children')],
    Input('main-plot', 'id'),  # Triggers on component mount
    [State('plot-type', 'value'),
     State('show-outliers', 'value')],
)
def initialize_plot(plot_id: str, plot_type: Optional[str], show_outliers: Optional[List[str]]) -> Tuple[go.Figure, go.Figure, html.Div]:
    from app import plot_df_3d
 
    if not plot_type:
        plot_type = '3d'
    
    include_outliers = show_outliers and 'outliers' in show_outliers
    
    df = plot_df_3d.copy()
    df.loc[:, 'enhanced_name'] = df.apply(create_enhanced_name, axis=1)
    
    if not include_outliers:
        df = df[df['cluster'] != -1]
    
    color_mapping = get_color_mapping(df)
    fig = create_base_plot(df, plot_type, color_mapping)
    barplot = create_cluster_barplot(df, color_mapping)
    
    return fig, barplot, html.Div()


@callback(
    [Output('main-plot', 'figure', allow_duplicate=True),
     Output('cluster-barplot', 'figure', allow_duplicate=True),
     Output('results-table', 'children', allow_duplicate=True)],
    Input('submit-btn', 'n_clicks'),
    [State('plot-type', 'value'),
     State('comparison-mode', 'value'),
     State({'type': 'search-dropdown', 'index': ALL}, 'value'),
     State('show-cluster-only', 'value'),
     State('show-outliers', 'value')],
    prevent_initial_call=True
)
def update_plot(n_clicks: int, plot_type: str, comparison_mode: List[str],
                selections: List[Optional[str]], cluster_only: List[str], 
                show_outliers: List[str]) -> Tuple[go.Figure, go.Figure, html.Div]:
    
    from app import plot_df_3d, plot_df_2d
    is_comparison = 'compare' in comparison_mode
    show_cluster = 'cluster_only' in cluster_only
    include_outliers = 'outliers' in show_outliers
    
    df = plot_df_3d.copy() if plot_type == '3d' else plot_df_2d.copy()
    
    df.loc[:, 'enhanced_name'] = df.apply(create_enhanced_name, axis=1)
    
    if not include_outliers:
        df = df[df['cluster'] != -1]
    
    color_mapping = get_color_mapping(df)
    
    # Filter out None selections
    valid_selections = [s for s in selections if s is not None]
    
    if is_comparison and len(valid_selections) >= 2:
        fig, results = create_comparison_plot(df, plot_type, valid_selections, 
                                     show_cluster, color_mapping)
    elif len(valid_selections) >= 1:
        fig, results = create_single_plot(df, plot_type, valid_selections[0], show_cluster, color_mapping)
    else:
        fig = create_base_plot(df, plot_type, color_mapping)
        results = html.Div()
    
    barplot = create_cluster_barplot(df, color_mapping)
    return fig, barplot, results

def create_enhanced_name(row: pd.Series) -> str:
    parts = [row['name']]
    if pd.notna(row['food_type_1']):
        parts.append(f"({row['food_type_1']}")
        if pd.notna(row['food_type_2']):
            parts.append(f"- {row['food_type_2']})")
        else:
            parts.append(")")
    elif pd.notna(row['food_type_2']):
        parts.append(f"({row['food_type_2']})")
    return " ".join(parts)


def parse_selection(selection: str) -> Tuple[Optional[str], Optional[str]]:
    if not selection:
        return None, None
    parts = selection.split(':', 1)
    if len(parts) == 2:
        search_type = 'name' if parts[0] == 'name' else 'food_type_1' if parts[0] == 'type1' else 'food_type_2'
        return parts[1], search_type
    return None, None


def get_matching_row(df: pd.DataFrame, value: str, search_type: str) -> pd.DataFrame:
    if search_type == 'name':
        matches = df[df['name'].str.lower() == value.lower()]
    elif search_type == 'food_type_1':
        matches = df[df['food_type_1'].str.lower() == value.lower()]
    else:
        matches = df[df['food_type_2'].str.lower() == value.lower()]
    return matches.head(1) if len(matches) > 0 else pd.DataFrame()



def create_base_plot(df: pd.DataFrame, plot_type: str, color_mapping: Dict[str, str]) -> go.Figure:
    df = df.copy()
    df.loc[:, 'cluster_str'] = df['cluster'].astype(str)
    
    if plot_type == '3d':
        fig = px.scatter_3d(df, x='UMAP1', y='UMAP2', z='UMAP3',
                           color='cluster_str', hover_name='enhanced_name',
                           color_discrete_map=color_mapping,
                           title='3DNuMAP')
        fig.update_traces(marker=dict(size=1, line=dict(width=0.2, color=COLORS['marker']['line'])))
    else:
        fig = px.scatter(df, x='UMAP1', y='UMAP2',
                        color='cluster_str', hover_name='enhanced_name',
                        color_discrete_map=color_mapping,
                        title='2DNuMAP')
        fig.update_traces(marker=dict(size=2, line=dict(width=0.2, color=COLORS['marker']['line'])))
    
    # Common layout updates
    fig.update_layout(
        height=PLOT_HEIGHT,
        plot_bgcolor=COLORS['background']['plot'],
        paper_bgcolor=COLORS['background']['paper'],
        font=dict(color=COLORS['primary']['main'], family="Arial, sans-serif"),
        
        margin=dict(l=50, r=50, t=60, b=50),
        showlegend=True,
        legend=dict(
            traceorder='reversed',
            bgcolor='rgba(58, 66, 73, 0.0)',
            borderwidth=0,
            font=dict(
                color=COLORS['primary']['main'],
                size=22
                ),
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor=COLORS['primary']['main'],
            mirror=True,
            gridcolor=COLORS['primary']['grid'],
            griddash='dash',
            zeroline=False,
            tickfont=dict(color=COLORS['primary']['main']),
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor=COLORS['primary']['main'],
            mirror=True,
            gridcolor=COLORS['primary']['grid'],
            griddash='dash',
            zeroline=False,
            tickfont=dict(color=COLORS['primary']['main']),
        ),
        
        title=dict(
            x=0.5,
            xanchor='center',
            font=dict(size=16, color=COLORS['primary']['main']),
            pad=dict(t=10, b=20)
        ),
    )
    
    if plot_type == '3d':
        fig.update_layout(
            scene=dict(
                bgcolor=COLORS['background']['scene'],
                xaxis=dict(
                    backgroundcolor=COLORS['background']['axis'],
                    gridcolor=COLORS['primary']['grid_3d'],
                    showgrid=True,
                    showbackground=True,
                    zerolinecolor=COLORS['primary']['zeroline'],
                    tickfont=dict(color=COLORS['primary']['main']),
                ),
                yaxis=dict(
                    backgroundcolor=COLORS['background']['axis'],
                    gridcolor=COLORS['primary']['grid_3d'],
                    showgrid=True,
                    showbackground=True,
                    zerolinecolor=COLORS['primary']['zeroline'],
                    tickfont=dict(color=COLORS['primary']['main']),
                ),
                zaxis=dict(
                    backgroundcolor=COLORS['background']['axis'],
                    gridcolor=COLORS['primary']['grid_3d'],
                    showgrid=True,
                    showbackground=True,
                    zerolinecolor=COLORS['primary']['zeroline'],
                    tickfont=dict(color=COLORS['primary']['main']),
                ),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.8),
                    up=dict(x=0, y=0, z=1),
                    projection=dict(type='perspective')
                )
            ),
            updatemenus=[
                {
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': 'Rotate',
                            'method': 'animate',
                            'args': [
                                None, 
                                {
                                    'frame': {'duration': 16, 'redraw': True},
                                    'fromcurrent': True,
                                    'mode': 'immediate',
                                    'transition': {'duration': 0, 'easing': 'linear'}
                                }
                            ]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [
                                [None],
                                {
                                    'mode': 'immediate',
                                    'frame': {'duration': 0, 'redraw': False},
                                    'transition': {'duration': 0}
                                }
                            ]
                        },
                        {
                            'label': 'Reset',
                            'method': 'relayout',
                            'args': [
                                {
                                    'scene.camera': dict(
                                        eye=dict(x=1.8, y=1.8, z=1.8),
                                        up=dict(x=0, y=0, z=1)
                                    )
                                }
                            ]
                        }
                    ],
                    'x': 0.02,
                    'y': 0.02,
                    'xanchor': 'left',
                    'yanchor': 'bottom',
                    'bgcolor': 'rgba(58, 66, 73, 0.9)',
                    'bordercolor': COLORS['primary']['main'],
                    'borderwidth': 1,
                    'font': {'color': COLORS['primary']['main'], 'size': 12},
                    'direction': 'up',
                    'pad': {'r': 10, 't': 10, 'b': 10}
                }
            ]
        )
        
        num_frames = 360   
        frames = []

        for i in range(num_frames):
            angle = i * (360.0 / num_frames)
            rad = np.radians(angle)
            
            base_radius = 1.8
            x_pos = base_radius * np.cos(rad)
            y_pos = base_radius * np.sin(rad)

            elliptical_factor = 0.95   
            x_pos *= elliptical_factor
            z_variation = 0.1 * np.sin(rad * 2)  
            z_pos = 1.7 + z_variation
            
            camera = dict(
                eye=dict(x=x_pos, y=y_pos, z=z_pos),
                up=dict(x=0, y=0, z=1)
            )
            
            frames.append(go.Frame(
                layout=dict(
                    scene_camera=camera,
                    title=f'3DNuMAP - Rotation {i+1}/{num_frames}'
                )
            ))

        fig.frames = frames
        
    return fig


def create_single_plot(df: pd.DataFrame, plot_type: str, selection: str,
                       show_cluster: bool, color_mapping: Dict[str, str]) -> Tuple[go.Figure, html.Div]:
    value, search_type = parse_selection(selection)
    if not value:
        return create_base_plot(df, plot_type, color_mapping), html.Div()
    
    selected = get_matching_row(df, value, search_type)
    if selected.empty:
        return create_base_plot(df, plot_type, color_mapping), html.Div("No matching food found")
    
    cluster_id = selected['cluster'].values[0]
    cluster_df = df[df['cluster'] == cluster_id].copy()
    
    coords_cols = ['UMAP1', 'UMAP2', 'UMAP3'] if plot_type == '3d' else ['UMAP1', 'UMAP2']
    coords = df[coords_cols].values
    selected_coords = selected[coords_cols].values[0]
    distances = np.linalg.norm(coords - selected_coords, axis=1)
    df.loc[:, 'distance'] = distances
    
    neighbors = df[df.index != selected.index[0]].nsmallest(NEAREST_NEIGHBORS, 'distance')
    
    base_df = cluster_df if show_cluster else df
    fig = create_base_plot(base_df, plot_type, color_mapping)
    
    if not show_cluster and cluster_id != -1:
        if plot_type == '3d':
            fig.add_trace(go.Scatter3d(
                x=cluster_df['UMAP1'], 
                y=cluster_df['UMAP2'], 
                z=cluster_df['UMAP3'],
                mode='markers',
                marker=dict(
                    size=8, 
                    color='rgba(0, 255, 65, 0.0)',
                    line=dict(width=1, color=COLORS['primary']['line'])
                ),
                hovertext=cluster_df['enhanced_name'],
                name=f'Cluster {cluster_id}',
                showlegend=True,
                hoverinfo='text'
            ))
        else:
            fig.add_trace(go.Scattergl(
                x=cluster_df['UMAP1'], 
                y=cluster_df['UMAP2'],
                mode='markers',
                marker=dict(
                    size=10, 
                    color='rgba(0, 255, 65, 0.0)',
                    line=dict(width=1, color=COLORS['primary']['line'])
                ),
                hovertext=cluster_df['enhanced_name'],
                name=f'Cluster {cluster_id}',
                showlegend=True,
                hoverinfo='text'
            ))
    
    selected_name = selected['enhanced_name'].values[0] if 'enhanced_name' in selected.columns else value
    
    if plot_type == '3d':
        fig.add_trace(go.Scatter3d(
            x=selected['UMAP1'], 
            y=selected['UMAP2'], 
            z=selected['UMAP3'],
            mode='markers+text',
            marker=dict(
                size=14, 
                color=COLORS['primary']['main'],
                symbol='diamond',
                line=dict(width=6, color=COLORS['primary']['border']),
                opacity=0.9
            ),
            text=[selected_name],
            textposition="top center",
            textfont=dict(color=COLORS['primary']['main'], size=12),
            name=f'Selected: {selected_name}',
            showlegend=True,
            hoverinfo='none'
        ))
    else:
        fig.add_trace(go.Scattergl(
            x=selected['UMAP1'], 
            y=selected['UMAP2'],
            mode='markers+text',
            marker=dict(
                size=16, 
                color=COLORS['primary']['main'],
                symbol='diamond',
                line=dict(width=6, color=COLORS['primary']['border']),
                opacity=0.9
            ),
            text=[selected_name],
            textposition="top center",
            textfont=dict(color=COLORS['primary']['main'], size=12),
            name=f'Selected: {selected_name}',
            showlegend=True,
            hoverinfo='none'
        ))
    
    table_data = neighbors[['enhanced_name', 'distance']].reset_index(drop=True)
    table = dbc.Table.from_dataframe(
        table_data.rename(columns={'enhanced_name': 'Food Name', 'distance': 'Distance'}),
        striped=True,
        bordered=False,
        hover=True,
        className="results-table mt-3"
    )

    results = html.Div([
        html.H5(
            f"Top {NEAREST_NEIGHBORS} foods similar to: {selected_name}",
            className="neon-title"
        ),
        html.P(
            f"Cluster: {cluster_id}",
            className="neon-subtitle"
        ),
        table
    ], className="results-container mt-4")

    return fig, results


def create_comparison_plot(df: pd.DataFrame, plot_type: str, selections: List[str],
                          show_cluster: bool, color_mapping: Dict[str, str]) -> Tuple[go.Figure, html.Div]:
    selected_foods = []
    cluster_ids = []
    
    for selection in selections:
        value, search_type = parse_selection(selection)
        if not value:
            continue
            
        selected = get_matching_row(df, value, search_type)
        if not selected.empty:
            selected_foods.append(selected)
            cluster_ids.append(selected['cluster'].values[0])
    
    if len(selected_foods) < 2:
        return create_base_plot(df, plot_type, color_mapping), html.Div("Need at least 2 valid foods for comparison")
    
    if show_cluster:
        base_df = df[df['cluster'].isin(cluster_ids)].copy()
    else:
        base_df = df.copy()
    
    coords_cols = ['UMAP1', 'UMAP2', 'UMAP3'] if plot_type == '3d' else ['UMAP1', 'UMAP2']
    coords = df[coords_cols].values
    
    neighbors_data = []
    for i, selected in enumerate(selected_foods):
        selected_coords = selected[coords_cols].values[0]
        df.loc[:, f'distance_{i}'] = np.linalg.norm(coords - selected_coords, axis=1)
        neighbors = df[df.index != selected.index[0]].nsmallest(NEAREST_NEIGHBORS, f'distance_{i}')
        neighbors_data.append(neighbors)
    
    fig = create_base_plot(base_df, plot_type, color_mapping)
    
    if plot_type == '3d':
        for i in range(len(selected_foods) - 1):
            fig.add_trace(go.Scatter3d(
                x=[selected_foods[i]['UMAP1'].values[0], selected_foods[i+1]['UMAP1'].values[0]],
                y=[selected_foods[i]['UMAP2'].values[0], selected_foods[i+1]['UMAP2'].values[0]],
                z=[selected_foods[i]['UMAP3'].values[0], selected_foods[i+1]['UMAP3'].values[0]],
                mode='lines',
                line=dict(
                    color=COLORS['comparison']['colors'][i % len(COLORS['comparison']['colors'])],
                    width=2,
                    dash='dash'
                ),
                name=f'Connection {i+1}-{i+2}',
                showlegend=True,
                hoverinfo='none'
            ))
    else:
        for i in range(len(selected_foods) - 1):
            fig.add_trace(go.Scattergl(
                x=[selected_foods[i]['UMAP1'].values[0], selected_foods[i+1]['UMAP1'].values[0]],
                y=[selected_foods[i]['UMAP2'].values[0], selected_foods[i+1]['UMAP2'].values[0]],
                mode='lines',
                line=dict(
                    color=COLORS['comparison']['colors'][i % len(COLORS['comparison']['colors'])],
                    width=2,
                    dash='dash'
                ),
                name=f'Connection {i+1}-{i+2}',
                showlegend=True,
                hoverinfo='none'
            ))
    
    for i, selected in enumerate(selected_foods):
        selected_name = selected['enhanced_name'].values[0] if 'enhanced_name' in selected.columns else selections[i].split(':', 1)[1]
        color_idx = i % len(COLORS['comparison']['colors'])
        border_color = COLORS['comparison']['borders'][color_idx]
        
        if plot_type == '3d':
            fig.add_trace(go.Scatter3d(
                x=selected['UMAP1'], 
                y=selected['UMAP2'], 
                z=selected['UMAP3'],
                mode='markers+text',
                marker=dict(
                    size=14, 
                    color=COLORS['comparison']['colors'][color_idx],
                    symbol='diamond',
                    line=dict(width=6, color=border_color),
                    opacity=0.9
                ),
                text=[f"Food {i+1}: {selected_name}"],
                textposition="top center",
                textfont=dict(color=COLORS['comparison']['colors'][color_idx], size=12),
                name=f'Food {i+1}: {selected_name}',
                showlegend=True,
                hoverinfo='none'
            ))
        else:
            fig.add_trace(go.Scattergl(
                x=selected['UMAP1'], 
                y=selected['UMAP2'],
                mode='markers+text',
                marker=dict(
                    size=16, 
                    color=COLORS['comparison']['colors'][color_idx],
                    symbol='diamond',
                    line=dict(width=6, color=border_color),
                    opacity=0.9
                ),
                text=[f"Food {i+1}: {selected_name}"],
                textposition="top center",
                textfont=dict(color=COLORS['comparison']['colors'][color_idx], size=12),
                name=f'Food {i+1}: {selected_name}',
                showlegend=True,
                hoverinfo='none'
            ))
    
    tables = []
    for i, (selected, neighbors) in enumerate(zip(selected_foods, neighbors_data)):
        selected_name = selected['enhanced_name'].values[0] if 'enhanced_name' in selected.columns else selections[i].split(':', 1)[1]

        table_data = neighbors[['enhanced_name', f'distance_{i}']].reset_index(drop=True)
        table = dbc.Table.from_dataframe(
            table_data.rename(columns={'enhanced_name': 'Food Name', f'distance_{i}': 'Distance'}),
            striped=True,
            bordered=False,
            hover=True,
            className="results-table mt-3"
        )
        
        tables.append(html.Div([
            html.H5(
                f"Food {i+1}: {selected_name}",
            ),
            html.P(f"Cluster: {cluster_ids[i]}"),
            table
        ], className="comparison-section mb-4"))
    
    results = html.Div([
        html.H4("Food Comparison Results", className="neon-title"),
        html.Div(tables, className="comparison-container")
    ], className="results-container mt-4")
    
    return fig, results