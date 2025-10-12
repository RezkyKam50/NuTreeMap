# utils/GAIAGX/tides.py
import requests
from datetime import date

def fetch_tide_stations(max_stations=None, start=None, end=None):
    stations_url = "https://surftruths.com/api/tide/stations.json"
    try:
        stations = requests.get(stations_url, timeout=10).json()
    except Exception as e:
        print(f"Error fetching station list: {e}")
        return []

    if max_stations:
        stations = stations[:max_stations]

    today = date.today().strftime("%Y%m%d")
    start = start or today
    end = end or today

    results = []

    for s in stations:
        try:
            sid = s["id"]
            pred_url = f"https://surftruths.com/api/tide/stations/{sid}/predictions.json?start={start}&end={end}"
            data = requests.get(pred_url, timeout=10).json()
            if not data:
                continue
 
            data.sort(key=lambda x: x["time"])
            upcoming = data[:2]
            info_text = "<br>".join(
                [f"{t['time']}: {t['type']} ({t['value']} cm)" for t in upcoming]
            )

            results.append({
                "name": s["name"],
                "lat": s["latitude"],
                "lon": s["longitude"],
                "predictions": info_text,
                "id": sid
            })
        except Exception as e:
            print(f"Error fetching tides for {s.get('name')}: {e}")
            continue

    return results
