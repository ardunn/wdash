from pyowm import OWM
from pyowm.commons.exceptions import InvalidSSLCertificateError
from pymongo import MongoClient
import logging
import pytz
from datetime import datetime
import time
import yaml
import os
import requests
import traceback
import sys

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(THIS_DIR, "config.yml"), "r") as config_file:
    CONFIG = yaml.safe_load(config_file)

HOST = CONFIG["DB_HOST"]
PORT = CONFIG["DB_PORT"]
NAME = CONFIG["DB_NAME"]
INTERVAL = CONFIG["INTERVAL"]
LAT = CONFIG["LAT"]
LON = CONFIG["LON"]
API_KEY = CONFIG["API_KEY"]

mc = MongoClient(host=HOST, port=PORT)
cw = mc[NAME].weather_current
cf = mc[NAME].weather_forecasted
ca = mc[NAME].aqi_current
caf = mc[NAME].aqi_forecasted
owm = OWM(API_KEY)
mgr = owm.weather_manager()
TIMEOUT_INTERVAL = 60          # seconds
handler = logging.StreamHandler()
formatter = logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)


def get_current_weather():
    """
    Get current weather as dictionary, with correct units converted.
    :return:
    """
    # r = requests.get(f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}")
    # w = r.json()

    w = mgr.weather_at_coords(LAT, LON).to_dict()["weather"]
    # w = mgr.weather_at_id(5403676).weather.to_dict()
    # print(w)
    w_converted = convert_units_weather_dict(w)
    return w_converted


def get_forecast_weathers():
    """
    Get forecasted weathers as list of dictionaries, with correct units converted.
    :return:
    """
    # r = requests.get(f"http://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}")
    # f = r.json()
    f = mgr.forecast_at_coords(LAT, LON, "3h").forecast.weathers
    f = [convert_units_weather_dict(w.to_dict()) for w in f]
    for i in f:
        # avoid fetched times being very close together but not identical
        i["fetched_at"] = f[0]["fetched_at"]
    return f


def get_air_pollution():
    r = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}")
    a = r.json()["list"][0]
    a = convert_format_air_pollution(a)
    return a


def get_forecast_air_pollution():
    r = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={API_KEY}")
    f = r.json()["list"]
    f = [convert_format_air_pollution(a) for a in f]
    # get every 10th entry as to not overwhelm db
    return f[::10]


def convert_units_weather_dict(weather):
    for k in ["sunset_time", "sunrise_time"]:
        if weather.get(k, None):
            weather[k] = local_timestamp2local_datetime(weather[k])
    weather["datetime"] = local_timestamp2local_datetime(weather["reference_time"])
    # convert from kelvin to F
    weather["temperature"] = {k: (T - 273.15) * 9/5 + 32 for k, T in weather["temperature"].items() if T}
    # convert from m/s to mph
    weather["wind"]["speed"] = weather["wind"]["speed"] * 2.237
    weather["fetched_at"] = datetime.now()
    return weather


def convert_format_air_pollution(air_pollution):
    d = air_pollution["components"]
    d.update(air_pollution["main"])
    d["datetime"] = local_timestamp2local_datetime(air_pollution["dt"])
    d["fetched_at"] = datetime.now()
    return d


def local_timestamp2local_datetime(timestamp):
    local_datetime = datetime.fromtimestamp(timestamp)
    return local_datetime


def datetime2string(datetime):
    return datetime.strftime("%m/%d/%Y, %H:%M:%S")


def main():
    while 1:
        try:
            w = get_current_weather()
            f = get_forecast_weathers()

            a = get_air_pollution()
            fa = get_forecast_air_pollution()
            logging.info("Current data and forecasts fetched")

            cw.insert_one(w)
            logging.info("Inserted current weather into DB")

            cf.insert_many(f)
            logging.info(f"Inserted {len(f)} forecasted weathers into DB")

            ca.insert_one(a)
            logging.info("Inserted current AQI into DB")

            caf.insert_many(fa)
            logging.info(f"Inserted {len(fa)} forecasted AQIs into DB")

            logging.info(f"Waiting {INTERVAL} seconds before next fetch")
            time.sleep(INTERVAL)
        except InvalidSSLCertificateError:
            logging.warning(f"Bad connection, waiting {TIMEOUT_INTERVAL} seconds...")
            time.sleep(TIMEOUT_INTERVAL)
        except BaseException:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tbf = traceback.format_exception(exc_type, exc_value, exc_traceback)
            logging.critical("Critical wdash daemon error:")
            logging.critical(tbf)


if __name__ == "__main__":
    main()


