from dash import Input, Output, State, callback
from utils.config.config import PLOT_HEIGHT, NEAREST_NEIGHBORS
from utils.Cmap import get_color_mapping
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dash import html

# Color Configuration
COLORS = {
    'background': {
        'plot': 'rgb(42, 50, 56)', # for 2d plot bg
        'paper': 'rgba(58, 66, 73, 0.1)',
        'scene': 'rgb(42, 50, 56)', # for 3d plot bg
        'axis': 'rgb(42, 50, 56)',
        'legend': 'rgb(58, 66, 73)'
    },
    'primary': {
        'main': '#00ffaf',
        'border': "#FAFFFB",
        'grid': 'rgba(0,255,65,0.1)',
        'grid_3d': 'rgba(100, 100, 100, 0.1)',
        'zeroline': 'rgba(0,255,65,0.5)',
        'highlight': 'rgba(0, 255, 65, 0.15)',
        'line': 'rgba(0, 255, 65, 0.5)'
    },
    'marker': {
        'line': 'rgba(0, 255, 175, 0.01)'
    },
    'comparison': {
        'color_1': '#00ffaf',
        'border_1': '#FFFFFF',
        'color_2': '#00ffaf',
        'border_2': '#FFFFFF'
    }
}

# Add callback for initial load
@callback(
    [Output('main-plot', 'figure'),
     Output('results-table', 'children')],
    Input('main-plot', 'id'),  # Triggers on component mount
    [State('plot-type', 'value'),
     State('show-outliers', 'value')],
)
def initialize_plot(plot_id: str, plot_type: Optional[str], show_outliers: Optional[List[str]]) -> Tuple[go.Figure, html.Div]:
    """Initialize the plot on dashboard launch with 3D view"""
    from app import plot_df_3d
    
    # Default to 3D plot
    if not plot_type:
        plot_type = '3d'
    
    include_outliers = show_outliers and 'outliers' in show_outliers
    
    df = plot_df_3d.copy()
    df.loc[:, 'enhanced_name'] = df.apply(create_enhanced_name, axis=1)
    
    if not include_outliers:
        df = df[df['cluster'] != -1]
    
    color_mapping = get_color_mapping(df)
    fig = create_base_plot(df, plot_type, color_mapping)
    
    return fig, html.Div()


@callback(
    [Output('main-plot', 'figure', allow_duplicate=True),
     Output('results-table', 'children', allow_duplicate=True)],
    Input('submit-btn', 'n_clicks'),
    [State('plot-type', 'value'),
     State('comparison-mode', 'value'),
     State('dropdown-1', 'value'),
     State('dropdown-2', 'value'),
     State('show-cluster-only', 'value'),
     State('show-outliers', 'value')],
    prevent_initial_call=True
)
def update_plot(n_clicks: int, plot_type: str, comparison_mode: List[str],
                selection_1: Optional[str], selection_2: Optional[str],
                cluster_only: List[str], show_outliers: List[str]) -> Tuple[go.Figure, html.Div]:
    
    from app import plot_df_3d, plot_df_2d
    is_comparison = 'compare' in comparison_mode
    show_cluster = 'cluster_only' in cluster_only
    include_outliers = 'outliers' in show_outliers
    
    df = plot_df_3d.copy() if plot_type == '3d' else plot_df_2d.copy()
    
    df.loc[:, 'enhanced_name'] = df.apply(create_enhanced_name, axis=1)
    
    if not include_outliers:
        df = df[df['cluster'] != -1]
    
    color_mapping = get_color_mapping(df)
    
    if is_comparison and selection_1 and selection_2:
        return create_comparison_plot(df, plot_type, selection_1, selection_2, 
                                     show_cluster, color_mapping)
    elif selection_1:
        return create_single_plot(df, plot_type, selection_1, show_cluster, color_mapping)
    
    fig = create_base_plot(df, plot_type, color_mapping)
    return fig, html.Div()


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
        fig.update_traces(marker=dict(size=3, line=dict(width=0.2, color=COLORS['marker']['line'])))
    else:
        fig = px.scatter(df, x='UMAP1', y='UMAP2',
                        color='cluster_str', hover_name='enhanced_name',
                        color_discrete_map=color_mapping,
                        title='2DNuMAP')
        fig.update_traces(marker=dict(size=5, line=dict(width=0.2, color=COLORS['marker']['line'])))
    
    # Common layout updates
    fig.update_layout(
        height=PLOT_HEIGHT,
        plot_bgcolor=COLORS['background']['plot'],
        paper_bgcolor=COLORS['background']['paper'],
        font=dict(color=COLORS['primary']['main'], family="Arial, sans-serif"),
        
        margin=dict(l=50, r=50, t=60, b=50),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(58, 66, 73, 0.0)',  # transparent, CSS handles style
            borderwidth=0,
            font=dict(color=COLORS['primary']['main']),
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
            # Move buttons inside the plot frame
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
                    # Position inside the plot (bottom left corner)
                    'x': 0.02,  # Left side
                    'y': 0.02,  # Bottom
                    'xanchor': 'left',
                    'yanchor': 'bottom',
                    'bgcolor': 'rgba(58, 66, 73, 0.9)',  # Semi-transparent background
                    'bordercolor': COLORS['primary']['main'],
                    'borderwidth': 1,
                    'font': {'color': COLORS['primary']['main'], 'size': 12},
                    'direction': 'up',  # Stack buttons vertically
                    'pad': {'r': 10, 't': 10, 'b': 10}  # Padding
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
                    color=COLORS['primary']['highlight'],
                    line=dict(width=0, color=COLORS['primary']['line'])
                ),
                hovertext=cluster_df['enhanced_name'],
                name=f'Cluster {cluster_id}',
                showlegend=True,
                hoverinfo='text'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=cluster_df['UMAP1'], 
                y=cluster_df['UMAP2'],
                mode='markers',
                marker=dict(
                    size=10, 
                    color=COLORS['primary']['highlight'],
                    line=dict(width=0, color=COLORS['primary']['line'])
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
        fig.add_trace(go.Scatter(
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
            className="neon-title"  # Use CSS class
        ),
        html.P(
            f"Cluster: {cluster_id}",
            className="neon-subtitle"  # Use CSS class
        ),
        table
    ], className="results-container mt-4")  # Container class handles the background/padding

    return fig, results


def create_comparison_plot(df: pd.DataFrame, plot_type: str, selection_1: str, selection_2: str,
                          show_cluster: bool, color_mapping: Dict[str, str]) -> Tuple[go.Figure, html.Div]:
    value_1, type_1 = parse_selection(selection_1)
    value_2, type_2 = parse_selection(selection_2)
    
    if not value_1 or not value_2:
        return create_base_plot(df, plot_type, color_mapping), html.Div()
    
    selected_1 = get_matching_row(df, value_1, type_1)
    selected_2 = get_matching_row(df, value_2, type_2)
    
    if selected_1.empty or selected_2.empty:
        return create_base_plot(df, plot_type, color_mapping), html.Div("One or both foods not found")
    
    cluster_1 = selected_1['cluster'].values[0]
    cluster_2 = selected_2['cluster'].values[0]
    
    if show_cluster:
        base_df = df[(df['cluster'] == cluster_1) | (df['cluster'] == cluster_2)].copy()
    else:
        base_df = df.copy()
    
    coords_cols = ['UMAP1', 'UMAP2', 'UMAP3'] if plot_type == '3d' else ['UMAP1', 'UMAP2']
    coords = df[coords_cols].values
    selected_coords_1 = selected_1[coords_cols].values[0]
    selected_coords_2 = selected_2[coords_cols].values[0]
    
    df.loc[:, 'distance_1'] = np.linalg.norm(coords - selected_coords_1, axis=1)
    df.loc[:, 'distance_2'] = np.linalg.norm(coords - selected_coords_2, axis=1)
    
    neighbors_1 = df[df.index != selected_1.index[0]].nsmallest(NEAREST_NEIGHBORS, 'distance_1')
    neighbors_2 = df[df.index != selected_2.index[0]].nsmallest(NEAREST_NEIGHBORS, 'distance_2')
    
    fig = create_base_plot(base_df, plot_type, color_mapping)
    
    selected_name_1 = selected_1['enhanced_name'].values[0] if 'enhanced_name' in selected_1.columns else value_1
    selected_name_2 = selected_2['enhanced_name'].values[0] if 'enhanced_name' in selected_2.columns else value_2
    
    selections = [
        (selected_1, COLORS['comparison']['color_1'], COLORS['comparison']['border_1'], selected_name_1, 'diamond'),
        (selected_2, COLORS['comparison']['color_2'], COLORS['comparison']['border_2'], selected_name_2, 'diamond')
    ]
    
    for selected, color, border_color, name, symbol in selections:
        if plot_type == '3d':
            fig.add_trace(go.Scatter3d(
                x=selected['UMAP1'], 
                y=selected['UMAP2'], 
                z=selected['UMAP3'],
                mode='markers+text',
                marker=dict(
                    size=14, 
                    color=color,
                    symbol=symbol,
                    line=dict(width=1, color=border_color),
                    opacity=0.9
                ),
                text=[name],
                textposition="top center",
                textfont=dict(color=color, size=12),
                name=f'Selected: {name}',
                showlegend=True,
                hoverinfo='none'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=selected['UMAP1'], 
                y=selected['UMAP2'],
                mode='markers+text',
                marker=dict(
                    size=16, 
                    color=color,
                    symbol=symbol,
                    line=dict(width=3, color=border_color),
                    opacity=0.9
                ),
                text=[name],
                textposition="top center",
                textfont=dict(color=color, size=12),
                name=f'Selected: {name}',
                showlegend=True,
                hoverinfo='none'
            ))
    
    table_1 = dbc.Table.from_dataframe(
        neighbors_1[['enhanced_name', 'distance_1']].reset_index(drop=True)
        .rename(columns={'enhanced_name': 'Food Name', 'distance_1': 'Distance'}),
        striped=True, bordered=False, hover=True, size='sm',
        className="results-table mt-2"
    )

    table_2 = dbc.Table.from_dataframe(
        neighbors_2[['enhanced_name', 'distance_2']].reset_index(drop=True)
        .rename(columns={'enhanced_name': 'Food Name', 'distance_2': 'Distance'}),
        striped=True, bordered=False, hover=True, size='sm',
        className="results-table mt-2"
    )

    results = dbc.Row([
        dbc.Col([
            html.H5(
                f"Similar to: {selected_name_1}",
                className="neon-title"  # Use CSS class instead of inline styles
            ),
            html.P(
                f"Cluster: {cluster_1}",
                className="neon-subtitle"  # Use CSS class instead of inline styles
            ),
            table_1
        ], width=6),
        dbc.Col([
            html.H5(
                f"Similar to: {selected_name_2}",
                className="neon-title"  # Use CSS class instead of inline styles
            ),
            html.P(
                f"Cluster: {cluster_2}",
                className="neon-subtitle"  # Use CSS class instead of inline styles
            ),
            table_2
        ], width=6)
    ], className="mt-4")
    
    return fig, results