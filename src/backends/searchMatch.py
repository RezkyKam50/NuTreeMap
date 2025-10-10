


def get_exact_match(plot_df, selected_name, search_type):
    if search_type == 'name':
        matches = plot_df[plot_df['name'].str.lower() == selected_name.lower()]
    elif search_type == 'food_type_1':
        matches = plot_df[plot_df['food_type_1'].str.lower() == selected_name.lower()]
    else:  
        matches = plot_df[plot_df['food_type_2'].str.lower() == selected_name.lower()]
     
    return matches.head(1) if len(matches) > 0 else matches