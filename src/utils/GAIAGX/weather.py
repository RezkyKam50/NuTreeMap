import openmeteo_requests
import requests_cache
from retry_requests import retry
from typing import Dict, List, Optional


cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_weather_data(latitude: float, longitude: float) -> Dict:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "weather_code"],
            "hourly": "temperature_2m",
            "timezone": "auto",
            "forecast_days": 1
        }
        
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Process current weather data
        current = response.Current()
        
        weather_data = {
            "temperature": current.Variables(0).Value(),
            "humidity": current.Variables(1).Value(),
            "precipitation": current.Variables(2).Value(),
            "weather_code": current.Variables(3).Value(),
            "elevation": response.Elevation(),
            "timezone": response.Timezone()
        }
        
        return weather_data
        
    except Exception as e:
        print(f"Error fetching weather data for {latitude}, {longitude}: {e}")
        return None

def get_weather_icon(weather_code: int) -> str:
    """
    Convert weather code to emoji icon
    """
    weather_icons = {
        0: "☀️",   # Clear sky
        1: "🌤️",   # Mainly clear
        2: "⛅",   # Partly cloudy
        3: "☁️",   # Overcast
        45: "🌫️",  # Fog
        48: "🌫️",  # Depositing rime fog
        51: "🌦️",  # Drizzle: Light
        53: "🌦️",  # Drizzle: Moderate
        55: "🌦️",  # Drizzle: Dense
        61: "🌧️",  # Rain: Slight
        63: "🌧️",  # Rain: Moderate
        65: "🌧️",  # Rain: Heavy
        80: "🌦️",  # Rain showers: Slight
        81: "🌦️",  # Rain showers: Moderate
        82: "🌦️",  # Rain showers: Violent
        95: "⛈️",   # Thunderstorm: Slight or moderate
        96: "⛈️",   # Thunderstorm with slight hail
        99: "⛈️",   # Thunderstorm with heavy hail
    }
    return weather_icons.get(weather_code, "❓")

def get_weather_description(weather_code: int) -> str:
    """
    Convert weather code to description
    """
    weather_descriptions = {
        0: "Clear sky",
        1: "Mainly clear", 
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle", 
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Heavy thunderstorm with hail",
    }
    return weather_descriptions.get(weather_code, "Unknown")


def get_major_cities_weather():
    
    major_cities = [
        # North America
        {"name": "New York", "lat": 40.71, "lon": -74.01},
        {"name": "Los Angeles", "lat": 34.05, "lon": -118.25},
        {"name": "Chicago", "lat": 41.88, "lon": -87.63},
        {"name": "Toronto", "lat": 43.65, "lon": -79.38},
        {"name": "Mexico City", "lat": 19.43, "lon": -99.13},
        {"name": "Vancouver", "lat": 49.28, "lon": -123.12},
        {"name": "Miami", "lat": 25.76, "lon": -80.19},
        {"name": "Montreal", "lat": 45.50, "lon": -73.57},
        {"name": "San Francisco", "lat": 37.77, "lon": -122.42},
        {"name": "Denver", "lat": 39.74, "lon": -104.99},

        # South America
        {"name": "São Paulo", "lat": -23.55, "lon": -46.63},
        {"name": "Buenos Aires", "lat": -34.60, "lon": -58.38},
        {"name": "Rio de Janeiro", "lat": -22.91, "lon": -43.17},
        {"name": "Lima", "lat": -12.04, "lon": -77.03},
        {"name": "Bogotá", "lat": 4.71, "lon": -74.07},
        {"name": "Santiago", "lat": -33.45, "lon": -70.65},
        {"name": "Montevideo", "lat": -34.90, "lon": -56.19},
        {"name": "Quito", "lat": -0.18, "lon": -78.47},
        {"name": "Caracas", "lat": 10.49, "lon": -66.88},
        {"name": "La Paz", "lat": -16.50, "lon": -68.12},

        # Europe
        {"name": "London", "lat": 51.51, "lon": -0.13},
        {"name": "Paris", "lat": 48.85, "lon": 2.35},
        {"name": "Berlin", "lat": 52.52, "lon": 13.41},
        {"name": "Rome", "lat": 41.90, "lon": 12.50},
        {"name": "Madrid", "lat": 40.42, "lon": -3.70},
        {"name": "Moscow", "lat": 55.76, "lon": 37.62},
        {"name": "Amsterdam", "lat": 52.37, "lon": 4.90},
        {"name": "Prague", "lat": 50.08, "lon": 14.43},
        {"name": "Vienna", "lat": 48.21, "lon": 16.37},
        {"name": "Dublin", "lat": 53.35, "lon": -6.26},
        {"name": "Lisbon", "lat": 38.72, "lon": -9.14},
        {"name": "Oslo", "lat": 59.91, "lon": 10.75},
        {"name": "Budapest", "lat": 47.50, "lon": 19.04},

        # Africa
        {"name": "Cairo", "lat": 30.04, "lon": 31.24},
        {"name": "Lagos", "lat": 6.46, "lon": 3.39},
        {"name": "Nairobi", "lat": -1.29, "lon": 36.82},
        {"name": "Johannesburg", "lat": -26.20, "lon": 28.04},
        {"name": "Accra", "lat": 5.56, "lon": -0.20},
        {"name": "Addis Ababa", "lat": 9.03, "lon": 38.74},
        {"name": "Casablanca", "lat": 33.57, "lon": -7.59},
        {"name": "Cape Town", "lat": -33.92, "lon": 18.42},

        # Asia
        {"name": "Tokyo", "lat": 35.69, "lon": 139.69},
        {"name": "Beijing", "lat": 39.90, "lon": 116.41},
        {"name": "Seoul", "lat": 37.57, "lon": 126.98},
        {"name": "Bangkok", "lat": 13.75, "lon": 100.50},
        {"name": "Mumbai", "lat": 19.08, "lon": 72.88},
        {"name": "Delhi", "lat": 28.61, "lon": 77.21},
        {"name": "Singapore", "lat": 1.35, "lon": 103.82},
        {"name": "Jakarta", "lat": -6.21, "lon": 106.85},
        {"name": "Kuala Lumpur", "lat": 3.14, "lon": 101.69},
        {"name": "Manila", "lat": 14.60, "lon": 120.98},

        # Oceania
        {"name": "Sydney", "lat": -33.87, "lon": 151.21},
        {"name": "Melbourne", "lat": -37.81, "lon": 144.96},
        {"name": "Auckland", "lat": -36.85, "lon": 174.76},
        {"name": "Wellington", "lat": -41.29, "lon": 174.78},
    ]

    
    cities_weather = []
    for city in major_cities:
        try:
            weather_data = fetch_weather_data(city["lat"], city["lon"])
            if weather_data:
                cities_weather.append({
                    "name": city["name"],
                    "lat": city["lat"],
                    "lon": city["lon"],
                    "temperature": weather_data["temperature"],
                    "humidity": weather_data["humidity"],
                    "precipitation": weather_data["precipitation"],
                    "weather_icon": get_weather_icon(weather_data["weather_code"]),
                    "weather_description": get_weather_description(weather_data["weather_code"])
                })
        except Exception as e:
            print(f"Error fetching weather for {city['name']}: {e}")
    
    return cities_weather
