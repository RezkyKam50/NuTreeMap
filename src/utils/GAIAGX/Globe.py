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
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.GAIAGX.weather import get_major_cities_weather
from utils.GAIAGX.population import fetch_live_population_data, fetch_rss_feeds

@callback(
    Output('earth-globe', 'figure'),
    [Input('region-selector', 'value'),
     Input('globe-mode', 'value'),]
)
def update_globe(region, mode):
    # Fetch live population data
    population_data = fetch_live_population_data()
    
    # Filter to include only countries with significant populations for better visualization
    significant_countries = {k: v for k, v in population_data.items() if v > 5}
    
    countries = list(significant_countries.keys())
    populations = list(significant_countries.values())
    
    df_globe = pd.DataFrame({
        'country': countries,
        'population': populations,
        'hover_text': [f"{c}: {p:,.1f} million people" for c, p in zip(countries, populations)]
    })
    
    # Fetch RSS feed data
    rss_data = fetch_rss_feeds()
    
    # Create the choropleth layer
    fig = go.Figure(data=go.Choropleth(
        locations=df_globe['country'],
        z=df_globe['population'],
        text=df_globe['hover_text'],
        colorscale='twilight',
        autocolorscale=False,
        reversescale=False,
        marker_line_width=0,
        colorbar_title="Population<br>(Millions)",
        colorbar=dict(
            bgcolor='rgba(26, 31, 36, 0.8)',
            tickfont=dict(color='#00ffaf')
        ),
        hoverinfo='text'
    ))
    
    # Add RSS feed markers - distributed globally
    if rss_data:
        rss_lats = [item['lat'] for item in rss_data]
        rss_lons = [item['lon'] for item in rss_data]
        rss_text = []
        
        for item in rss_data:
            base_text = f"<b>{item['title']}</b><br>{item['published']}<br>{item['summary']}"
            
            # Add weather information if available
            if 'temperature' in item:
                weather_text = (
                    f"<br><br>🌡️ <b>Current Weather:</b><br>"
                    f"{item['weather_icon']} {item['weather_description']}<br>"
                    f"Temperature: {item['temperature']}<br>"
                    f"Humidity: {item['humidity']}<br>"
                    f"Precipitation: {item['precipitation']}"
                )
                base_text += weather_text
            
            base_text += f"<br><a href='{item['link']}' target='_blank'>Read more</a>"
            rss_text.append(base_text)
        
        fig.add_trace(go.Scattergeo(
            lon=rss_lons,
            lat=rss_lats,
            text=rss_text,
            mode='markers',
            hoverinfo='text',
            name='News Feed',
            showlegend=True,
            marker=dict(
                size=8,
                color='#00ffaf',
                symbol='circle'
            )
        ))
    
    # Add weather data for major cities if toggle is enabled
        cities_weather = get_major_cities_weather()
        
        if cities_weather:
            city_lats = [city['lat'] for city in cities_weather]
            city_lons = [city['lon'] for city in cities_weather]
            city_text = [
                f"<b>{city['name']}</b><br>"
                f"{city['weather_icon']} {city['weather_description']}<br>"
                f"🌡️ Temperature: {city['temperature']:.1f}°C<br>"
                f"💧 Humidity: {city['humidity']:.0f}%<br>"
                f"🌧️ Precipitation: {city['precipitation']:.1f}mm"
                for city in cities_weather
            ]
            
            fig.add_trace(go.Scattergeo(
                lon=city_lons,
                lat=city_lats,
                text=city_text,
                mode='markers',
                hoverinfo='text',
                name='Weather Stations',
                showlegend=True,
                marker=dict(
                    size=15,
                    color='#ff6b6b',
                    symbol='arrow',
                    line=dict(width=1, color='white')
                )
            ))
    
    # Projection mapping
    projection_map = {
        'geo': 'equirectangular',
        'ortho': 'orthographic',
        'natural': 'natural earth'
    }
    
    # Region center coordinates
    region_coords = {
        'NA': {'lat': 40, 'lon': -100},
        'SA': {'lat': -15, 'lon': -60},
        'EU': {'lat': 50, 'lon': 10},
        'AF': {'lat': 0, 'lon': 20},
        'AS': {'lat': 30, 'lon': 100},
        'OC': {'lat': -25, 'lon': 135}
    }
    
    coords = region_coords.get(region, {'lat': 0, 'lon': 0})
    
    # Update geos
    fig.update_geos(
        projection_type=projection_map.get(mode, 'natural earth'),
        showland=True,
        landcolor='#2A3238',
        oceancolor='#1a1f24',
        showocean=True,
        showcountries=False,
        projection_rotation=dict(lon=coords['lon'], lat=coords['lat']),
        bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_layout(
        height=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#00ffaf', size=12),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            x=0,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(26, 31, 36, 0.8)',
            font=dict(color='#00ffaf')
        )
    )
    
    return fig

 