import dash_bootstrap_components as dbc, plotly.express as px, plotly.graph_objects as go, pandas as pd, numpy as np
from typing import Dict, List, Tuple, Optional, Any



def get_color_mapping(df: pd.DataFrame) -> Dict[str, str]:
    unique_clusters = sorted(df['cluster'].unique())
    
    # Elegant muted palette with good distinction
    elegant_colors = [
        '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
        '#6B8E23', '#956BB0', '#00A8E8', '#FF6B6B', '#4ECDC4',
        '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA',
        '#F1948A', '#85C1E9', '#D7BDE2', '#F9E79F', '#A9DFBF',
        '#F5B7B1', '#AED6F1', '#D2B4DE', '#FAD7A0'
    ]
    
    return {str(c): elegant_colors[i % len(elegant_colors)] for i, c in enumerate(unique_clusters)}