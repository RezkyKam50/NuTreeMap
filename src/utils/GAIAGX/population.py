import feedparser
from datetime import datetime
import random
import requests
from typing import Dict, List, Optional
from utils.GAIAGX.weather import fetch_weather_data, get_weather_description, get_weather_icon


def fetch_live_population_data() -> Dict[str, float]:
    try:
        url = "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=300&date=2023"
        
        response = requests.get(url, timeout=10)
        data = response.json()

        population_data = {}
        
        if len(data) > 1 and 'data' in data[1]:
            for item in data[1]:
                if item['value'] is not None:
                    country_code = item['country']['id']
                    population_millions = item['value'] / 1_000_000   
                    population_data[country_code] = round(population_millions, 1)
        
        if not population_data:
            return fetch_backup_population_data()
            
        return population_data
        
    except Exception as e:
        print(f"Error fetching World Bank population data: {e}")
        return fetch_backup_population_data()

def fetch_backup_population_data() -> Dict[str, float]:
    try:
        url = "https://restcountries.com/v3.1/all?fields=cca3,population"
        response = requests.get(url, timeout=10)
        countries = response.json()
        
        population_data = {}
        for country in countries:
            country_code = country.get('cca3', '')
            population = country.get('population', 0)
            if population > 0:
                population_millions = population / 1_000_000
                population_data[country_code] = round(population_millions, 1)
        
        return population_data
        
    except Exception as e:
        print(f"Error fetching backup population data: {e}")
        return get_minimal_fallback_data()

def get_minimal_fallback_data() -> Dict[str, float]:
    """Minimal fallback data for critical countries"""
    return {
        'USA': 335, 'CHN': 1425, 'IND': 1428, 'IDN': 277, 'PAK': 231,
        'BRA': 216, 'NGA': 218, 'BGD': 171, 'RUS': 144, 'MEX': 128,
        'JPN': 123, 'ETH': 123, 'PHL': 115, 'EGY': 109, 'VNM': 98,
        'COD': 95, 'TUR': 85, 'IRN': 88, 'DEU': 84, 'THA': 71,
        'GBR': 68, 'FRA': 68, 'ITA': 59, 'ZAF': 60, 'KOR': 52,
        'COL': 52, 'ESP': 48, 'ARG': 46, 'UKR': 37, 'POL': 38,
        'CAN': 39, 'AUS': 26, 'NLD': 17, 'BEL': 12, 'SWE': 10
    }

def get_global_coordinates():
    regions = [
        # North America
        {'lat_range': (25, 55), 'lon_range': (-130, -70)},
        # South America
        {'lat_range': (-35, 10), 'lon_range': (-80, -35)},
        # Europe
        {'lat_range': (35, 65), 'lon_range': (-10, 40)},
        # Africa
        {'lat_range': (-35, 35), 'lon_range': (-20, 50)},
        # Asia
        {'lat_range': (10, 60), 'lon_range': (70, 140)},
        # Oceania
        {'lat_range': (-45, -10), 'lon_range': (110, 180)},
    ]
    
    region = random.choice(regions)
    
    # Generate random coordinates within that region
    lat = random.uniform(*region['lat_range'])
    lon = random.uniform(*region['lon_range'])
    
    return lat, lon

def fetch_rss_feeds():
    feeds = [
        'http://rss.nytimes.com/services/xml/rss/nyt/Health.xml',
        'https://www.sciencedaily.com/rss/health_medicine/nutrition.xml',
    ]
    
    feed_data = []
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:  
                lat, lon = get_global_coordinates()

                weather_data = fetch_weather_data(lat, lon)
                
                feed_entry = {
                    'title': entry.get('title', 'No title'),
                    'link': entry.get('link', '#'),
                    'published': entry.get('published', 'Unknown date'),
                    'summary': entry.get('summary', 'No summary')[:200] + '...',
                    'lat': lat,
                    'lon': lon
                }

                if weather_data:
                    feed_entry.update({
                        'temperature': f"{weather_data['temperature']:.1f}°C",
                        'humidity': f"{weather_data['humidity']:.0f}%",
                        'precipitation': f"{weather_data['precipitation']:.1f}mm",
                        'weather_icon': get_weather_icon(weather_data['weather_code']),
                        'weather_description': get_weather_description(weather_data['weather_code'])
                    })
                
                feed_data.append(feed_entry)
        except Exception as e:
            print(f"Error fetching feed {feed_url}: {e}")
    
    return feed_data