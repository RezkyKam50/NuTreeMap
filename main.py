import streamlit as st, plotly.express as px, pandas as pd, numpy as np
from ollama import chat   
from xgb_infer import (load_models_and_components, 
                       predict_with_loaded_models
)

PLOT_CONFIG = {
    # General Plot Settings
    'height': 800,
    'dark_theme': False,
    
    'cluster_colors': px.colors.qualitative.Set3,   
    'background_color': 'black',
    'grid_color': 'gray',
    'font_color': 'white',
    
    # Marker Settings - Base Points
    'base_marker_size_3d': 3,
    'base_marker_size_2d': 5,
    'base_marker_line_width': 1,
    'base_marker_line_color': 'black',
    
    # Marker Settings - Highlighted Points
    'highlight_marker_size_3d': 20,
    'highlight_marker_size_2d': 20,
    'highlight_marker_line_width_3d': 2,
    'highlight_marker_line_width_2d': 3,
    'highlight_marker_line_color': 'black',
    
    # Marker Settings - Cluster Highlights
    'cluster_marker_size_3d': 20,
    'cluster_marker_size_2d': 20,
    'cluster_marker_opacity': 0.3,
    'cluster_marker_color': 'red',
    'cluster_marker_line_width_3d': 0.3,
    'cluster_marker_line_width_2d': 0.3,
    
    # Selection Colors
    'food1_color': 'green',    # Food 1 in comparison mode
    'food2_color': 'green',   # Food 2 in comparison mode
    'single_food_color': 'green',  # Single food selection
    
    # Search and Display Settings
    'max_name_suggestions': 20,
    'max_type1_suggestions': 10,
    'max_type2_suggestions': 10,
    'nearest_neighbors_count': 10,
    
    # Default Values
    'default_food1': 'Beef',
    'default_food2': 'Chicken',
    'default_single_food': 'Beef'
}

HOVER_DATA_CONFIG = {
    'UMAP1': False,
    'UMAP2': False,
    'UMAP3': False,  # Only used in 3D mode
    'cluster': False,
    'calories': True,
    'total_fat': True,
    'saturated_fat': True,
    'cholesterol': True,
    'sodium': True,
    'food_type_1': True,
    'food_type_2': True,
    'enhanced_name': False
}

# =============================================================================
# END CONFIGURATION PARAMETERS
# =============================================================================

@st.cache_data
def load_data():
    return pd.read_parquet("umapped3D.parquet", engine='pyarrow')

@st.cache_data
def load_2d():
    return pd.read_parquet("umapped2D.parquet", engine='pyarrow')

@st.cache_resource
def load_prediction_models():
    return load_models_and_components()

def get_cluster_color_mapping(plot_df):
    unique_clusters = sorted(plot_df['cluster'].unique())
    
    all_colors = (
        px.colors.qualitative.Set3 + 
        px.colors.qualitative.Dark2 + 
        px.colors.qualitative.Set1 + 
        px.colors.qualitative.Pastel1 + 
        px.colors.qualitative.Pastel2 +
        px.colors.qualitative.Safe
    )
    
    color_mapping = {}
    for i, cluster in enumerate(unique_clusters):
        color_mapping[cluster] = all_colors[i % len(all_colors)]
    
    return color_mapping

plot_df_3d = load_data()
plot_df_2d = load_2d()

all_names = plot_df_3d['name'].dropna().unique()
all_food_types_1 = plot_df_3d['food_type_1'].dropna().unique()
all_food_types_2 = plot_df_3d['food_type_2'].dropna().unique()

st.title("NuTreeMap")

plot_type = st.radio("Select visualization type:", ["3D UMAP", "2D UMAP"], horizontal=True)

comparison_mode = st.checkbox("Enable food comparison mode", value=False)

def create_search_section(label, key_prefix, default_value=None):
    if default_value is None:
        default_value = PLOT_CONFIG['default_single_food']
    
    typed_input = st.text_input(f"Search food DB (by name or food type) - {label}", 
                               value=default_value, key=f"{key_prefix}_input")

    suggestions = []

    name_matches = [name for name in all_names if typed_input.lower() in name.lower()]
    suggestions.extend([(name, 'name') for name in name_matches[:PLOT_CONFIG['max_name_suggestions']]])

    type1_matches = [ftype for ftype in all_food_types_1 if typed_input.lower() in ftype.lower()]
    suggestions.extend([(ftype, 'food_type_1') for ftype in type1_matches[:PLOT_CONFIG['max_type1_suggestions']]])

    type2_matches = [ftype for ftype in all_food_types_2 if typed_input.lower() in ftype.lower()]
    suggestions.extend([(ftype, 'food_type_2') for ftype in type2_matches[:PLOT_CONFIG['max_type2_suggestions']]])

    if suggestions:
        formatted_suggestions = []
        for item, item_type in suggestions:
            if item_type == 'name':
                formatted_suggestions.append(f"🍽️ {item}")
            elif item_type == 'food_type_1':
                formatted_suggestions.append(f"📂 Type 1: {item}")
            else:   
                formatted_suggestions.append(f"🏷️ Type 2: {item}")
        
        selected_suggestion = st.selectbox(f"Result - {label}:", 
                                         formatted_suggestions, key=f"{key_prefix}_suggestions")
        
        if selected_suggestion:
            if selected_suggestion.startswith("🍽️ "):
                selected_name = selected_suggestion[3:]   
                search_type = 'name'
            elif selected_suggestion.startswith("📂 Type 1: "):
                selected_name = selected_suggestion[11:]   
                search_type = 'food_type_1'
            elif selected_suggestion.startswith("🏷️ Type 2: "):
                selected_name = selected_suggestion[11:]   
                search_type = 'food_type_2'
        else:
            selected_name = None
            search_type = None
    else:
        selected_name = None
        search_type = None
    
    return selected_name, search_type

if comparison_mode:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Food 1")
        selected_name_1, search_type_1 = create_search_section("Food 1", "food1", PLOT_CONFIG['default_food1'])
    
    with col2:
        st.subheader("Food 2")
        selected_name_2, search_type_2 = create_search_section("Food 2", "food2", PLOT_CONFIG['default_food2'])
    
    show_cluster_only = st.checkbox("Show clusters for selected foods only", value=False)
else:
    selected_name, search_type = create_search_section("Food", "single", PLOT_CONFIG['default_single_food'])
    show_cluster_only = st.checkbox(f"Show current cluster for {selected_name if selected_name else 'selection'}", value=False)

show_outliers = st.checkbox("Show outliers (cluster = -1)", value=True)
submit = st.button("Submit")
st.caption("Clustering are done purely by nutritional content and not by food name and food type")
def get_exact_match(plot_df, selected_name, search_type):
    if search_type == 'name':
        matches = plot_df[plot_df['name'].str.lower() == selected_name.lower()]
    elif search_type == 'food_type_1':
        matches = plot_df[plot_df['food_type_1'].str.lower() == selected_name.lower()]
    else:  
        matches = plot_df[plot_df['food_type_2'].str.lower() == selected_name.lower()]
     
    return matches.head(1) if len(matches) > 0 else matches

def get_hover_data(plot_type):
    hover_data = HOVER_DATA_CONFIG.copy()
    if plot_type == "2D UMAP":
        hover_data.pop('UMAP3', None)   
    return hover_data

def apply_plot_styling(fig, plot_type):
    if PLOT_CONFIG['dark_theme']:
        fig.update_layout(
            paper_bgcolor=PLOT_CONFIG['background_color'],
            plot_bgcolor=PLOT_CONFIG['background_color'],
            font=dict(color=PLOT_CONFIG['font_color'])
        )
        
        if plot_type == "3D UMAP":
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        backgroundcolor=PLOT_CONFIG['background_color'], 
                        gridcolor=PLOT_CONFIG['grid_color'], 
                        showbackground=True, 
                        zerolinecolor=PLOT_CONFIG['grid_color'], 
                        color=PLOT_CONFIG['font_color']
                    ),
                    yaxis=dict(
                        backgroundcolor=PLOT_CONFIG['background_color'], 
                        gridcolor=PLOT_CONFIG['grid_color'], 
                        showbackground=True, 
                        zerolinecolor=PLOT_CONFIG['grid_color'], 
                        color=PLOT_CONFIG['font_color']
                    ),
                    zaxis=dict(
                        backgroundcolor=PLOT_CONFIG['background_color'], 
                        gridcolor=PLOT_CONFIG['grid_color'], 
                        showbackground=True, 
                        zerolinecolor=PLOT_CONFIG['grid_color'], 
                        color=PLOT_CONFIG['font_color']
                    )
                )
            )
        else:
            fig.update_layout(
                xaxis=dict(
                    gridcolor=PLOT_CONFIG['grid_color'], 
                    zerolinecolor=PLOT_CONFIG['grid_color'], 
                    color=PLOT_CONFIG['font_color']
                ),
                yaxis=dict(
                    gridcolor=PLOT_CONFIG['grid_color'], 
                    zerolinecolor=PLOT_CONFIG['grid_color'], 
                    color=PLOT_CONFIG['font_color']
                )
            )

def create_base_plot(base_df, plot_type, color_mapping):
    hover_data_common = get_hover_data(plot_type)
    base_df = base_df.copy()
    base_df['cluster_str'] = base_df['cluster'].astype(str)
    
    if plot_type == "3D UMAP":
        fig = px.scatter_3d(
            base_df,
            height=PLOT_CONFIG['height'],
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            color='cluster_str',
            hover_name='enhanced_name',
            hover_data=hover_data_common,
            title='3D Food Clusters with UMAP',
            color_discrete_map={str(k): v for k, v in color_mapping.items()}
        )
        fig.update_traces(marker=dict(
            size=PLOT_CONFIG['base_marker_size_3d'], 
            line=dict(width=PLOT_CONFIG['base_marker_line_width'], color=PLOT_CONFIG['base_marker_line_color'])
        ))
    else:
        fig = px.scatter(
            base_df,
            height=PLOT_CONFIG['height'],
            x='UMAP1',
            y='UMAP2',
            color='cluster_str',
            hover_name='enhanced_name',
            hover_data=hover_data_common,
            title='2D Food Clusters with UMAP',
            color_discrete_map={str(k): v for k, v in color_mapping.items()}
        )
        fig.update_traces(marker=dict(
            size=PLOT_CONFIG['base_marker_size_2d'], 
            line=dict(width=PLOT_CONFIG['base_marker_line_width'], color=PLOT_CONFIG['base_marker_line_color'])
        ))
    
    return fig

if submit:
    plot_df = plot_df_3d if plot_type == "3D UMAP" else plot_df_2d
    
    color_mapping = get_cluster_color_mapping(plot_df)
    
    def create_hover_name(row):
        name = row['name']
        type1 = row['food_type_1'] if pd.notna(row['food_type_1']) else ''
        type2 = row['food_type_2'] if pd.notna(row['food_type_2']) else ''
        
        hover_parts = [name]
        if type1:
            hover_parts.append(f"({type1}")
            if type2:
                hover_parts.append(f"- {type2})")
            else:
                hover_parts.append(")")
        elif type2:
            hover_parts.append(f"({type2})")
            
        return " ".join(hover_parts)
    
    plot_df = plot_df.copy()
    plot_df['enhanced_name'] = plot_df.apply(create_hover_name, axis=1)
    if not show_outliers:
        plot_df = plot_df[plot_df['cluster'] != -1]

    if comparison_mode:
        if selected_name_1 and selected_name_2:
            selected_row_1 = get_exact_match(plot_df, selected_name_1, search_type_1)
            selected_row_2 = get_exact_match(plot_df, selected_name_2, search_type_2)
            
            if len(selected_row_1) > 0 and len(selected_row_2) > 0:
                selected_cluster_1 = selected_row_1['cluster'].values[0]
                selected_cluster_2 = selected_row_2['cluster'].values[0]
                
                cluster_mask_1 = plot_df['cluster'] == selected_cluster_1
                cluster_mask_2 = plot_df['cluster'] == selected_cluster_2
                
                if show_cluster_only:
                    base_df = plot_df[cluster_mask_1 | cluster_mask_2]
                else:
                    base_df = plot_df
                if plot_type == "3D UMAP":
                    coords = plot_df[['UMAP1', 'UMAP2', 'UMAP3']].values
                    selected_coords_1 = selected_row_1[['UMAP1', 'UMAP2', 'UMAP3']].values[0]
                    selected_coords_2 = selected_row_2[['UMAP1', 'UMAP2', 'UMAP3']].values[0]
                else:
                    coords = plot_df[['UMAP1', 'UMAP2']].values
                    selected_coords_1 = selected_row_1[['UMAP1', 'UMAP2']].values[0]
                    selected_coords_2 = selected_row_2[['UMAP1', 'UMAP2']].values[0]

                distances_1 = np.linalg.norm(coords - selected_coords_1, axis=1)
                distances_2 = np.linalg.norm(coords - selected_coords_2, axis=1)
                
                plot_df['distance_1'] = distances_1
                plot_df['distance_2'] = distances_2
                exact_mask_1 = (plot_df.index == selected_row_1.index[0])
                exact_mask_2 = (plot_df.index == selected_row_2.index[0])
                
                nearest_neighbors_1 = plot_df[~exact_mask_1].nsmallest(PLOT_CONFIG['nearest_neighbors_count'], 'distance_1')[['enhanced_name', 'distance_1']]
                nearest_neighbors_2 = plot_df[~exact_mask_2].nsmallest(PLOT_CONFIG['nearest_neighbors_count'], 'distance_2')[['enhanced_name', 'distance_2']]

                fig = create_base_plot(base_df, plot_type, color_mapping)
                hover_data_common = get_hover_data(plot_type)

                if plot_type == "3D UMAP":
                    fig.add_trace(
                        px.scatter_3d(
                            selected_row_1,
                            x='UMAP1',
                            y='UMAP2',
                            z='UMAP3',
                            hover_name='enhanced_name',
                            hover_data=hover_data_common
                        ).update_traces(
                            marker=dict(
                                size=PLOT_CONFIG['highlight_marker_size_3d'], 
                                color=PLOT_CONFIG['food1_color'], 
                                symbol='circle-open', 
                                line=dict(width=PLOT_CONFIG['highlight_marker_line_width_3d'], color=PLOT_CONFIG['highlight_marker_line_color'])
                            ),
                            showlegend=False,
                            name=f'Food 1: {selected_name_1}'
                        ).data[0]
                    )

                    fig.add_trace(
                        px.scatter_3d(
                            selected_row_2,
                            x='UMAP1',
                            y='UMAP2',
                            z='UMAP3',
                            hover_name='enhanced_name',
                            hover_data=hover_data_common
                        ).update_traces(
                            marker=dict(
                                size=PLOT_CONFIG['highlight_marker_size_3d'], 
                                color=PLOT_CONFIG['food2_color'], 
                                symbol='circle-open', 
                                line=dict(width=PLOT_CONFIG['highlight_marker_line_width_3d'], color=PLOT_CONFIG['highlight_marker_line_color'])
                            ),
                            showlegend=False,
                            name=f'Food 2: {selected_name_2}'
                        ).data[0]
                    )
                else:
                    fig.add_trace(
                        px.scatter(
                            selected_row_1,
                            x='UMAP1',
                            y='UMAP2',
                            hover_name='enhanced_name',
                            hover_data=hover_data_common
                        ).update_traces(
                            marker=dict(
                                size=PLOT_CONFIG['highlight_marker_size_2d'], 
                                color=PLOT_CONFIG['food1_color'], 
                                symbol='circle-open', 
                                line=dict(width=PLOT_CONFIG['highlight_marker_line_width_2d'], color=PLOT_CONFIG['highlight_marker_line_color'])
                            ),
                            showlegend=False,
                            name=f'Food 1: {selected_name_1}'
                        ).data[0]
                    )
                    fig.add_trace(
                        px.scatter(
                            selected_row_2,
                            x='UMAP1',
                            y='UMAP2',
                            hover_name='enhanced_name',
                            hover_data=hover_data_common
                        ).update_traces(
                            marker=dict(
                                size=PLOT_CONFIG['highlight_marker_size_2d'], 
                                color=PLOT_CONFIG['food2_color'], 
                                symbol='circle-open', 
                                line=dict(width=PLOT_CONFIG['highlight_marker_line_width_2d'], color=PLOT_CONFIG['highlight_marker_line_color'])
                            ),
                            showlegend=False,
                            name=f'Food 2: {selected_name_2}'
                        ).data[0]
                    )

                apply_plot_styling(fig, plot_type)
                st.plotly_chart(fig, use_container_width=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Top {PLOT_CONFIG['nearest_neighbors_count']} foods similar to: {selected_name_1}")
                    st.table(nearest_neighbors_1.reset_index(drop=True).rename(columns={'enhanced_name': 'Food Name', 'distance_1': 'Distance'}))
                
                with col2:
                    st.subheader(f"Top {PLOT_CONFIG['nearest_neighbors_count']} foods similar to: {selected_name_2}")
                    st.table(nearest_neighbors_2.reset_index(drop=True).rename(columns={'enhanced_name': 'Food Name', 'distance_2': 'Distance'}))

            else:
                if len(selected_row_1) == 0:
                    st.error(f"No data found for {selected_name_1} in {search_type_1.replace('_', ' ')}")
                if len(selected_row_2) == 0:
                    st.error(f"No data found for {selected_name_2} in {search_type_2.replace('_', ' ')}")
        else:
            st.warning("Please select both foods for comparison mode.")
    
    else:
        if selected_name:
            selected_row = get_exact_match(plot_df, selected_name, search_type)
            
            if len(selected_row) > 0:
                selected_cluster = selected_row['cluster'].values[0]
                cluster_mask = plot_df['cluster'] == selected_cluster
                cluster_df = plot_df[cluster_mask]
                if plot_type == "3D UMAP":
                    coords = plot_df[['UMAP1', 'UMAP2', 'UMAP3']].values
                    selected_coords = selected_row[['UMAP1', 'UMAP2', 'UMAP3']].values[0]
                else:
                    coords = plot_df[['UMAP1', 'UMAP2']].values
                    selected_coords = selected_row[['UMAP1', 'UMAP2']].values[0]

                distances = np.linalg.norm(coords - selected_coords, axis=1)
                plot_df['distance'] = distances
                
                exact_mask = (plot_df.index == selected_row.index[0])
                nearest_neighbors = plot_df[~exact_mask].nsmallest(PLOT_CONFIG['nearest_neighbors_count'], 'distance')[['enhanced_name', 'distance']]

                base_df = cluster_df if show_cluster_only else plot_df

                fig = create_base_plot(base_df, plot_type, color_mapping)
                hover_data_common = get_hover_data(plot_type)

                if not show_cluster_only:
                    if plot_type == "3D UMAP":
                        fig.add_trace(
                            px.scatter_3d(
                                cluster_df,
                                height=PLOT_CONFIG['height'],
                                x='UMAP1',
                                y='UMAP2',
                                z='UMAP3',
                                hover_name='enhanced_name',
                                hover_data=hover_data_common
                            ).update_traces(
                                marker=dict(
                                    size=PLOT_CONFIG['cluster_marker_size_3d'], 
                                    color=PLOT_CONFIG['cluster_marker_color'], 
                                    opacity=PLOT_CONFIG['cluster_marker_opacity'], 
                                    symbol='circle-open', 
                                    line=dict(width=PLOT_CONFIG['cluster_marker_line_width_3d'], color=PLOT_CONFIG['base_marker_line_color'])
                                ),
                                showlegend=False
                            ).data[0]
                        )
                    else:
                        fig.add_trace(
                            px.scatter(
                                cluster_df,
                                height=PLOT_CONFIG['height'],
                                x='UMAP1',
                                y='UMAP2',
                                hover_name='enhanced_name',
                                hover_data=hover_data_common
                            ).update_traces(
                                marker=dict(
                                    size=PLOT_CONFIG['cluster_marker_size_2d'], 
                                    color=PLOT_CONFIG['cluster_marker_color'], 
                                    opacity=PLOT_CONFIG['cluster_marker_opacity'], 
                                    symbol='circle-open', 
                                    line=dict(width=PLOT_CONFIG['cluster_marker_line_width_2d'], color=PLOT_CONFIG['base_marker_line_color'])
                                ),
                                showlegend=False
                            ).data[0]
                        )
                if plot_type == "3D UMAP":
                    fig.add_trace(
                        px.scatter_3d(
                            selected_row,
                            height=PLOT_CONFIG['height'],
                            x='UMAP1',
                            y='UMAP2',
                            z='UMAP3',
                            hover_name='enhanced_name',
                            hover_data=hover_data_common
                        ).update_traces(
                            marker=dict(
                                size=PLOT_CONFIG['highlight_marker_size_3d'], 
                                color=PLOT_CONFIG['single_food_color'], 
                                symbol='circle-open', 
                                line=dict(width=PLOT_CONFIG['highlight_marker_line_width_3d'], color=PLOT_CONFIG['highlight_marker_line_color'])
                            ),
                            showlegend=False
                        ).data[0]
                    )
                else:
                    fig.add_trace(
                        px.scatter(
                            selected_row,
                            height=PLOT_CONFIG['height'],
                            x='UMAP1',
                            y='UMAP2',
                            hover_name='enhanced_name',
                            hover_data=hover_data_common
                        ).update_traces(
                            marker=dict(
                                size=PLOT_CONFIG['highlight_marker_size_2d'], 
                                color=PLOT_CONFIG['single_food_color'], 
                                symbol='circle-open', 
                                line=dict(width=PLOT_CONFIG['highlight_marker_line_width_2d'], color=PLOT_CONFIG['highlight_marker_line_color'])
                            ),
                            showlegend=False
                        ).data[0]
                    )

                apply_plot_styling(fig, plot_type)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader(f"Top {PLOT_CONFIG['nearest_neighbors_count']} foods similar to: {selected_name} ({search_type.replace('_', ' ').title()})")
                st.table(nearest_neighbors.reset_index(drop=True).rename(columns={'enhanced_name': 'Food Name', 'distance': 'Distance'}))
            else:
                st.error(f"No data found for {selected_name} in {search_type.replace('_', ' ')}")
        else:
            st.warning("Please select a food item.")

if not submit:
    if comparison_mode:
        st.write("Select two foods and click Submit to compare them on the plot.")
    else:
        st.write("Enter a food name or food type and click Submit to see the plot and nearest neighbors.")

with st.expander("Nutri-XG", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Food Information")
        food_name = st.text_input("Input food name or type or name and its type", value="Potato, Salted")
        st.caption("Syntax : [Meat/Liquids][Preparation][Seasoning]")

        st.subheader("Known Nutritional Values")
        st.caption("Enter known values. Leave others as 0 if unknown.")

        protein = st.number_input("Protein (g)", min_value=0.0, value=25.0, step=0.1)
        fat = st.number_input("Total Fat (g)", min_value=0.0, value=3.0, step=0.1)
        carbs = st.number_input("Carbohydrates (g)", min_value=0.0, value=120.0, step=0.1)
        sodium = st.number_input("Sodium (g)", min_value=0.0, value=20.0, step=0.1)
        cholesterol = st.number_input("Cholesterol (g)", min_value=0.0, value=10.0, step=0.1)

        predict_btn = st.button("🔮 Predict Nutrients", type="primary")

    with col2:
        st.subheader("Prediction Results")

        if predict_btn:
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

                with st.spinner("Predicting..."):
                    results = predict_with_loaded_models(
                        food_name=food_name,
                        numeric_values=numeric_values,
                        loaded_components=load_prediction_models()
                    )
                model_type = "Text-Only Model (2 nutrients)" if numeric_values is None else "Mixed Model (13 nutrients)"
                st.success(f"✅ Used {model_type} for: **{results['food_name']}**")
                if results['model1_predictions']:
                    st.subheader("XGBoost : Text (TF-IDF + Numerical)")
                    st.caption("7 nutrients predicted")
                    model1_data = [
                        {
                            'Nutrient': k.replace('_', ' ').title(),
                            'Predicted Value': f"{max(v, 0):.2f}",
                            'Unit': (
                                'kJ' if k == 'calories' else
                                'g' if k in [
                                    'saturated_fat', 'fiber', 'protein', 'total_fat', 'carbohydrate',
                                    'monounsaturated_fatty_acids', 'saturated_fatty_acids', 'polyunsaturated_fatty_acids'
                                ] else
                                'g'
                            )
                        }
                        for k, v in results['model1_predictions'].items()
                        if k != 'calcium'
                    ]
                    st.dataframe(model1_data, hide_index=True)
                if results['model2_predictions'] and numeric_values is None:
                    st.subheader("XGBoost : Text (TF-IDF)")
                    st.caption("2 nutrients predicted")
                    model2_data = [
                        {
                            'Nutrient': k.replace('_', ' ').title(),
                            'Predicted Value': f"{max(v, 0):.2f}",
                            'Unit': 'g'
                        }
                        for k, v in results['model2_predictions'].items()
                        if k in ['protein', 'carbohydrate']
                    ]
                    st.dataframe(model2_data, hide_index=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)

        else:
            st.info("👆 Click 'Predict Nutrients' to begin")
            st.markdown("""
            - Leave all fields at 0 to use the **text-only model**
            - Enter any known nutrients to use the **mixed model**
            """)
st.markdown("---")