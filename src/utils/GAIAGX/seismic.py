# seismic_live.py
import json
import logging
from tornado.websocket import websocket_connect
from tornado.ioloop import IOLoop
from tornado import gen
from collections import deque

echo_uri = 'wss://www.seismicportal.eu/standing_order/websocket'
PING_INTERVAL = 10
 
recent_events = deque(maxlen=1000)

@gen.coroutine
def listen(ws):
    while True:
        msg = yield ws.read_message()
        if msg is None:
            logging.info("WebSocket closed.")
            break

        try:
            data = json.loads(msg)
            props = data['data']['properties']
            coords = data['data']['geometry']['coordinates']
            lon, lat, depth = coords
            mag = props.get('mag', None)
            region = props.get('flynn_region', 'Unknown')

            recent_events.append({
                'lat': lat,
                'lon': lon,
                'depth': depth,
                'mag': mag,
                'region': region,
                'time': props.get('time'),
            })
        except Exception:
            logging.exception("Error parsing message")

@gen.coroutine
def launch_client():
    logging.info("Connecting to Seismic Portal...")
    ws = yield websocket_connect(echo_uri, ping_interval=PING_INTERVAL)
    logging.info("Listening for earthquake updates...")
    listen(ws)

def start_seismic_listener():
    ioloop = IOLoop()
    ioloop.spawn_callback(launch_client)
    ioloop.start()
