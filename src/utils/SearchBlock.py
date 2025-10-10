from dash import Input, Output, callback
from utils.config.config import MAX_SUGGESTIONS
from typing import Dict, List, Tuple

# Don't import from app here - we'll get data from the callback context

# Callbacks
@callback(
    [Output('search-section-1', 'style'),
     Output('search-section-2', 'style')],
    Input('comparison-mode', 'value')
)
def toggle_search_sections(comparison_mode: List[str]) -> Tuple[Dict, Dict]:
    is_comparison = 'compare' in comparison_mode
    
    if is_comparison:
        return {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'block'}, {'display': 'none'}


@callback(
    Output('dropdown-1', 'options'),
    Input('search-1', 'value')
)
def update_dropdown_1(search_text: str) -> List[Dict[str, str]]:
    # Import inside function to avoid circular import
    from app import all_names, all_food_types_1, all_food_types_2
    return get_search_suggestions(search_text, all_names, all_food_types_1, all_food_types_2)


@callback(
    Output('dropdown-2', 'options'),
    Input('search-2', 'value')
)
def update_dropdown_2(search_text: str) -> List[Dict[str, str]]:
    # Import inside function to avoid circular import
    from app import all_names, all_food_types_1, all_food_types_2
    return get_search_suggestions(search_text, all_names, all_food_types_1, all_food_types_2)


def get_search_suggestions(search_text: str, all_names: list, all_food_types_1: list, all_food_types_2: list) -> List[Dict[str, str]]:
    if not search_text:
        return []
    
    suggestions = []
    search_lower = search_text.lower()
    
    # Name matches
    name_matches = [n for n in all_names if search_lower in n.lower()][:MAX_SUGGESTIONS]
    suggestions.extend([{'label': f"🍽️ {n}", 'value': f"name:{n}"} for n in name_matches])
    
    # Type 1 matches
    type1_matches = [t for t in all_food_types_1 if search_lower in t.lower()][:10]
    suggestions.extend([{'label': f"📂 Type 1: {t}", 'value': f"type1:{t}"} for t in type1_matches])
    
    # Type 2 matches
    type2_matches = [t for t in all_food_types_2 if search_lower in t.lower()][:10]
    suggestions.extend([{'label': f"🏷️ Type 2: {t}", 'value': f"type2:{t}"} for t in type2_matches])
    
    return suggestions