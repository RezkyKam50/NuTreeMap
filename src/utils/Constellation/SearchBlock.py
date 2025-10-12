from dash import Input, Output, callback
from utils.config.config import MAX_SUGGESTIONS
from typing import Dict, List, Tuple
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
import dash_bootstrap_components as dbc
import dash


@callback(
    [Output({'type': 'search-section', 'index': 0}, 'style'),
     Output({'type': 'search-section', 'index': 1}, 'style')],
    Input('comparison-mode', 'value')
)
def toggle_search_sections(comparison_mode: List[str]) -> Tuple[Dict, Dict]:
    is_comparison = 'compare' in comparison_mode
    
    if is_comparison:
        return {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'block'}, {'display': 'none'}

def get_search_suggestions(search_text: str, all_names: list, all_food_types_1: list, all_food_types_2: list) -> List[Dict[str, str]]:
    if not search_text:
        return []
    
    suggestions = []
    search_lower = search_text.lower()

    name_matches = [n for n in all_names if search_lower in n.lower()][:MAX_SUGGESTIONS]
    suggestions.extend([{'label': f"🍽️ {n}", 'value': f"name:{n}"} for n in name_matches])

    type1_matches = [t for t in all_food_types_1 if search_lower in t.lower()][:10]
    suggestions.extend([{'label': f"📂 Type 1: {t}", 'value': f"type1:{t}"} for t in type1_matches])

    type2_matches = [t for t in all_food_types_2 if search_lower in t.lower()][:10]
    suggestions.extend([{'label': f"🏷️ Type 2: {t}", 'value': f"type2:{t}"} for t in type2_matches])
    
    return suggestions



def update_dropdown(search_value: str) -> List[Dict[str, str]]:

    from app import all_names, all_food_types_1, all_food_types_2
    
    if not search_value or len(search_value) < 2:
        return []
    
    search_lower = search_value.lower()   
    options = []

    name_matches = [name for name in all_names if search_lower in name.lower()]
    for name in name_matches[:MAX_SUGGESTIONS]:
        options.append({
            'label': f"🍽️ {name}",
            'value': f"name:{name}"
        })

    type1_matches = [ft for ft in all_food_types_1 if search_lower in ft.lower()]
    for ft in type1_matches[:MAX_SUGGESTIONS]:
        options.append({
            'label': f"🔖 {ft}",
            'value': f"type1:{ft}"
        })

    type2_matches = [ft for ft in all_food_types_2 if search_lower in ft.lower()]
    for ft in type2_matches[:MAX_SUGGESTIONS]:
        options.append({
            'label': f"📂 {ft}",
            'value': f"type2:{ft}"
        })
    
    return options[:MAX_SUGGESTIONS]


 
@callback(
    Output('search-sections-container', 'children'),
    Output('food-controls', 'style'),
    Input('comparison-mode', 'value'),
    Input('add-food-btn', 'n_clicks'),
    Input('remove-food-btn', 'n_clicks'),
    State('search-sections-container', 'children')
)
def update_search_sections(comparison_mode, add_clicks, remove_clicks, current_children):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_children, {'display': 'none'}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    is_comparison = 'compare' in comparison_mode
    
    if not current_children:
        current_children = [create_search_section(0)]
    
    controls_style = {'display': 'block'} if is_comparison else {'display': 'none'}
    
    if trigger_id == 'comparison-mode':
        if is_comparison:
            if len(current_children) < 2:
                current_children = [create_search_section(0), create_search_section(1)]
        else:
            current_children = [create_search_section(0)]
    
    elif trigger_id == 'add-food-btn' and is_comparison:
        new_index = len(current_children)
        current_children.append(create_search_section(new_index))
    
    elif trigger_id == 'remove-food-btn' and is_comparison and len(current_children) > 2:
        current_children = current_children[:-1]
    
    return current_children, controls_style

def create_search_section(index):
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.H5(f"Food {index + 1}", style={'color': '#00ffaf'}),
                dbc.Input(
                    id={'type': 'search-input', 'index': index}, 
                    placeholder="Search food...", 
                    value=" " if index == 0 else "", 
                    className="mb-2",
                    style={
                        'backgroundColor': '#3A4249',
                        'color': '#00ffaf',
                        'borderColor': '#00ffaf',
                        'borderWidth': '1px'
                    }
                ),
                dcc.Dropdown(
                    id={'type': 'search-dropdown', 'index': index}, 
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
