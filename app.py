import datetime

from pymongo import MongoClient
import yaml
import os
from functools import reduce
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import logging
import sys
import traceback

handler = logging.StreamHandler()
formatter = logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(THIS_DIR, "config.yml"), "r") as config_file:
    CONFIG = yaml.safe_load(config_file)

HOST = CONFIG["DB_HOST"]
PORT = CONFIG["DB_PORT"]
NAME = CONFIG["DB_NAME"]
INTERVAL = CONFIG["INTERVAL"]
SERVER_PORT = CONFIG["PORT"]
WINDY_EMBED = CONFIG["WINDY_EMBED"]


mc = MongoClient(host=HOST, port=PORT)
cw = mc[NAME].weather_current
cf = mc[NAME].weather_forecasted
ca = mc[NAME].aqi_current
caf = mc[NAME].aqi_forecasted

N_FORECASTS_SHOWN = 10
N_FORECAST_WINDOW = 7
FORECAST_WINDOW_UNITS = "d"
N_WIND_STAR_WINDOW = 5
WIND_STAR_WINDOW_UNITS = "h"
FORECAST_RGB = "rgb(160,160,160)"
N_HISTORICAL_DAYS = 7

SUMMARY_FORECAST_DAYS = 7

# based on the "main" columns from https://openweathermap.org/weather-conditions
OWM_WEATHER_SIMPLE_STATUS_TO_INT = {
    "clear": 0,
    "clouds": 1,
    "drizzle": 2,
    "rain": 3,
    "snow": 10,
    "mist": 4,
    "fog": 5,
    "sand": 8,
    "smoke": 20,
    "haze": 15,
    "dust": 13,
    "ash": 25,
    "squall": 17,
    "tornado": 30
}

app = dash.Dash(__name__, title="wdash", update_title=None)


def convert_meteorological_deg2cardinal_dir(deg_measurement):
    """
    from
    http://snowfence.umn.edu/Components/winddirectionanddegrees.htm
    :param deg_measurement:
    :return:
    """

    if deg_measurement > 348.75 or deg_measurement <= 11.25:
        return "N"
    elif deg_measurement > 11.25 and deg_measurement <= 33.25:
        return "NNE"
    elif deg_measurement > 33.75 and deg_measurement <= 56.25:
        return "NE"
    elif deg_measurement > 56.25 and deg_measurement <= 78.75:
        return "ENE"
    elif deg_measurement > 78.75 and deg_measurement <= 101.25:
        return "E"
    elif deg_measurement > 101.25 and deg_measurement <= 123.75:
        return "ESE"
    elif deg_measurement > 123.75 and deg_measurement <= 146.25:
        return "SE"
    elif deg_measurement > 146.25 and deg_measurement <= 168.75:
        return "S"
    elif deg_measurement > 168.75 and deg_measurement <= 191.25:
        return "SSW"
    elif deg_measurement > 191.25 and deg_measurement <= 213.75:
        return "SSW"
    elif deg_measurement > 213.75 and deg_measurement <= 236.25:
        return "SW"
    elif deg_measurement > 236.25 and deg_measurement <= 258.75:
        return "WSW"
    elif deg_measurement > 258.75 and deg_measurement <= 281.25:
        return "W"
    elif deg_measurement > 281.25 and deg_measurement <= 303.75:
        return "WNW"
    elif deg_measurement > 303.75 and deg_measurement <= 326.25:
        return "NW"
    elif deg_measurement > 326.25 and deg_measurement <= 348.75:
        return "NNW"

def recursive_get(d, *keys):
    """
    Taken from https://stackoverflow.com/questions/28225552/is-there-a-recursive-version-of-the-dict-get-built-in
    :param d:
    :param keys:
    :return:
    """
    return reduce(lambda c, k: c.get(k, {}), keys, d)


def datetime2string(d_series):
    return [d.strftime("%m/%d/%Y, %H:%M:%S") for d in d_series]

def fetch_history():
    return [i for i in cw.find({})]

def fetch_forecasts():
    return [i for i in cf.find({})]

def fetch_aqi_history():
    return [i for i in ca.find({})]

def fetch_aqi_forecasts():
    return [i for i in caf.find({})]

def weathers2df(weathers):
    """
    Compile a list of dict weathers into a dataframe

    :param weathers:
    """
    # get comprehensive list of keys, assuming max depth of 2
    keys_flat = []
    for w in weathers:
        for k, v in w.items():
            if isinstance(v, dict):
                for kn, vn in v.items():
                    key_flat = ".".join((k, kn))
                    keys_flat.append(key_flat)
            else:
                keys_flat.append(k)
    keys_flat = sorted(list(set(keys_flat)))
    keys_args = [tuple(k.split(".")) for k in keys_flat]
    data = {k: [np.nan] * len(weathers) for k in keys_flat}
    for i, w in enumerate(weathers):
        for j, k in enumerate(keys_args):
            v = recursive_get(w, *k)
            key_flat = keys_flat[j]
            if v:
                data[key_flat][i] = v
    return pd.DataFrame(data)


def create_df():
    f = fetch_forecasts()
    fdf = weathers2df(f)
    fdf["origin"] = "forecast"
    h = fetch_history()
    hdf = weathers2df(h)
    hdf["origin"] = "history"
    dt_cols = ["datetime", "fetched_at"]
    dt_cols_str = [d + "_str" for d in dt_cols]
    df = pd.concat((hdf, fdf))
    df[dt_cols_str] = df[dt_cols].apply(datetime2string)
    df["wind.cardinal"] = df["wind.deg"].apply(convert_meteorological_deg2cardinal_dir)
    df["sunset.timeofday"] = df["sunset_time"].apply(lambda x: x.time() if isinstance(x, datetime.datetime) else x )
    df["sunrise.timeofday"] = df["sunrise_time"].apply(lambda x: x.time() if isinstance(x, datetime.datetime) else x)

    # manually fix a few columns
    for c in ["precipitation_probability", "snow.3h", "rain.3h"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
        else:
            logging.info(f"Expected column '{c}' not found in dataframe!")

    df = df.reset_index()

    return df


def create_aqi_df():
    f = fetch_aqi_forecasts()
    afdf = weathers2df(f)
    afdf["origin"] = "forecast"

    h = fetch_aqi_history()
    ahdf = weathers2df(h)
    ahdf["origin"] = "history"

    adf = pd.concat((ahdf, afdf))

    dt_cols = ["datetime", "fetched_at"]
    dt_cols_str = [d + "_str" for d in dt_cols]
    adf[dt_cols_str] = adf[dt_cols].apply(datetime2string)

    adf["aqi"] = adf["aqi"].apply(lambda x: {1: "great", 2: "fair", 3: "moderate", 4: "poor", 5: "hazardous"}[x])

    return adf


def get_error_graphs(df_graph, x_column, main_line_column, lower_column, upper_column, rgb_str, name):
    line = go.Scatter(
        x=df_graph[x_column],
        y=df_graph[main_line_column],
        mode="lines",
        line=dict(color=rgb_str),
        marker=dict(color=rgb_str),
        name=name
    )
    rgba_str = rgb_str.replace("rgb", "rgba").replace(")", ",0.2)")
    rgba_nothing = 'rgba(0,0,0,0)'
    area = go.Scatter(
        x=df_graph[x_column].tolist() + df_graph[x_column].tolist()[::-1],
        y=df_graph[upper_column].tolist() + df_graph[lower_column].tolist()[::-1],
        fill="toself",
        fillcolor=rgba_str,
        line=dict(color=rgba_nothing),
        marker=dict(color=rgba_nothing),
        showlegend=False,
        hoverinfo="skip"
    )
    return [line, area]


def get_line_graphs(df_graph, x_column, line_column, rgba_str, line_width, mode, name):
    line = go.Scatter(
        x=df_graph[x_column],
        y=df_graph[line_column],
        mode=mode,
        line=dict(color=rgba_str, width=line_width),
        marker=dict(color=rgba_str),
        name=name
    )
    return [line]



def get_n_most_recent_forecast_series(df_forecast, column_main, window, window_unit):
    if window_unit not in ("h", "d"):
        raise ValueError("Invalid window_unit, must be 'h' for hours or 'd' for days.")

    now = datetime.datetime.now()
    timedeltas = now - df_forecast["datetime"]
    is_in_forecast_display_window = timedeltas <= pd.Timedelta(window, unit=window_unit)
    df_forecast2show = df_forecast[is_in_forecast_display_window].reset_index()


    # Choose a number of forecasts to show
    forecast2show_fetchstrs = df_forecast2show["fetched_at_str"].unique()
    # Ensure the forecast series will be spaced out across all the unique fetched at times
    # if there are more forecasts than will be shown for the given time period
    if len(forecast2show_fetchstrs) > N_FORECASTS_SHOWN:
        logging.warning(
            f"Number of forecasts to show ({N_FORECASTS_SHOWN}) for '{column_main}' is smaller than number of available forecasts ({len(forecast2show_fetchstrs)}) in time window of {window} {'days' if window_unit == 'd' else 'hours'}")
        fetched_at_2show = np.ceil(
            np.linspace(0, len(forecast2show_fetchstrs), N_FORECASTS_SHOWN, endpoint=False)).astype(int)
        forecast2show_fetchstrs = forecast2show_fetchstrs[fetched_at_2show]

    transparencies = np.linspace(0.1, 1, N_FORECASTS_SHOWN)

    return forecast2show_fetchstrs, transparencies


def create_time_figure(
        df,
        column_main,
        title,
        show_forecast=True,
        show_history=True,
        column_error_min=None,
        column_error_max=None,
        history_rgb="rgb(255,0,0)",
        forecast_rgb="rgb(0,0,0)",
        as_type="figure",
        line_only_graph_type="lines",
        custom_main_trace_label=None,
        limit_date_range=True
):
    df_forecast = df[df["origin"] == "forecast"]
    df_history = df[df["origin"] == "history"]

    graphs = []

    main_trace_label = "Recorded data" if not custom_main_trace_label else custom_main_trace_label

    if show_history:
        if column_error_max and column_error_min:
            logging.info(f"Creating '{column_main}' graph with historical error bars")
            history_graphs = get_error_graphs(
                df_history,
                "datetime",
                column_main,
                column_error_min,
                column_error_max,
                history_rgb,
                main_trace_label
            )
        else:
            logging.info(f"Creating '{column_main}' graph with no error bars")
            history_graphs = get_line_graphs(
                df_history,
                "datetime",
                column_main,
                history_rgb,
                3,
                line_only_graph_type,
                main_trace_label
            )
        graphs += history_graphs

    if show_forecast:
        forecast2show_fetchstrs, transparencies = get_n_most_recent_forecast_series(df_forecast, column_main, N_FORECAST_WINDOW, FORECAST_WINDOW_UNITS)

        forecast_graphs = []
        for i, fetched_time in enumerate(forecast2show_fetchstrs):
            df_fetched_at = df_forecast[df_forecast["fetched_at_str"] == fetched_time]
            forecast_graph = get_line_graphs(
                df_fetched_at,
                "datetime",
                column_main,
                forecast_rgb.replace("rgb", "rgba").replace(")", f",{transparencies[i]})"),
                1,
                "lines",
                f"Forecast @ {fetched_time}"
            )
            forecast_graphs += forecast_graph
        graphs += forecast_graphs

    if as_type == "figure":
        fig = go.Figure(graphs)
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=column_main,
            legend_title="Data Source",
            template="plotly_dark"
        )
        if limit_date_range:
            limited_td = datetime.timedelta(days=N_HISTORICAL_DAYS)
            if datetime.datetime.now() - df_history["datetime"].min() > limited_td:
                dt_min = datetime.datetime.now() - limited_td
            else:
                dt_min = df_history["datetime"].min()

            dt_max = df_forecast["datetime"].max() if show_forecast else df_history["datetime"].max()
            fig.update_xaxes(range=[dt_min, dt_max])
        return fig
    elif as_type == "graphs":
        return graphs
    else:
        raise TypeError(f"as_type '{as_type}' is not a valid type!")


def wind_direction_graph(df_wind, history_rgb, forecast_rgb):
    wind_key = "wind.cardinal"
    df_wind = df_wind.dropna(subset=[wind_key])
    df_history = df_wind[df_wind["origin"] == "history"]
    df_forecast = df_wind[df_wind["origin"] == "forecast"]
    hovertemplate = 'time:%{customdata} <br>speed:%{r:.3f}<br>direction:%{theta}'

    fig = go.Figure()

    # only use data from the past hour
    timedeltas = datetime.datetime.now() - df_history["datetime"]
    is_in_display_window = timedeltas <= pd.Timedelta(N_WIND_STAR_WINDOW, unit=WIND_STAR_WINDOW_UNITS)
    df_history = df_history[is_in_display_window]

    history_polar = go.Scatterpolar(
        r=df_history["wind.speed"],
        theta=df_history[wind_key],
        mode="lines+markers",
        name="Recorded data",
        line=dict(color=history_rgb, width=3),
        marker=dict(color=history_rgb),
        customdata=df_history["fetched_at_str"],
        hovertemplate=hovertemplate
    )


    # get most recent one and highlight it

    ix_most_recent = timedeltas.idxmin()

    most_recent = go.Scatterpolar(
        r=[df_history["wind.speed"].loc[ix_most_recent]],
        theta=[df_history[wind_key].loc[ix_most_recent]],
        mode="markers",
        marker=dict(color="rgb(255,0,0)", size=10),
        name="Most recent recording",
        customdata=[df_history["fetched_at_str"].loc[ix_most_recent]],
        hovertemplate='time:%{customdata} <br>speed:%{r:.3f}<br>direction:%{theta}'
    )


    forecast2show_fetchstrs, transparencies = get_n_most_recent_forecast_series(df_forecast, wind_key, N_WIND_STAR_WINDOW, WIND_STAR_WINDOW_UNITS)
    for i, fetched_time in enumerate(forecast2show_fetchstrs):
        df_fetched_at = df_forecast[df_forecast["fetched_at_str"] == fetched_time]
        forecast_graph = go.Scatterpolar(
            r=df_fetched_at["wind.speed"],
            theta=df_fetched_at[wind_key],
            mode="lines",
            name=f"Forecast @ {fetched_time}",
            line=dict(color=forecast_rgb.replace("rgb", "rgba").replace(")", f",{transparencies[i]})"), width=1),
            visible="legendonly",
            customdata=df_fetched_at["datetime_str"],
            hovertemplate=hovertemplate
        )
        fig.add_trace(forecast_graph)

    fig.add_trace(history_polar)
    fig.add_trace(most_recent)

    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            angularaxis=dict(
                categoryarray=["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"],
                categoryorder="array",
                type="category",
                direction="clockwise"
            ),
        ),
        title="Wind direction/speed star (radial mph, angular direction)",
        legend_title="Data source"
    )
    return fig


@app.callback(
    Output(component_id='all_info', component_property="children"),
    Input(component_id="interval", component_property="n_intervals")
)
def generate_page(n_intervals):
    try:
        logging.info(f"Regenerating info div: times regenerated = {n_intervals}")
        df = create_df()

        adf = create_aqi_df()

        fig_temp = create_time_figure(
            df,
            "temperature.temp",
            "Temperature (F)",
            show_forecast=True,
            show_history=True,
            column_error_min="temperature.temp_min",
            column_error_max="temperature.temp_max",
            history_rgb="rgb(255,0,0)",
            forecast_rgb=FORECAST_RGB
        )

        fig_temp_historical = create_time_figure(
            df,
            "temperature.temp",
            "Temperature (F) - All time",
            show_forecast=False,
            show_history=True,
            history_rgb="rgb(255,0,0)",
            limit_date_range=False,
        )


        fig_precipitation_prob = create_time_figure(
            df,
            "precipitation_probability",
            "Probability of precipitation",
            show_forecast=True,
            show_history=True,
            history_rgb="rgb(0,0,255)",
            forecast_rgb=FORECAST_RGB
        )

        fig_windspeed = create_time_figure(
            df,
            "wind.speed",
            "Wind speed (mph)",
            show_forecast=True,
            show_history=True,
            history_rgb="rgb(0,255,0)",
            forecast_rgb=FORECAST_RGB
        )
        fig_windspeed_historical = create_time_figure(
            df,
            "wind.speed",
            "Wind speed (mph) - All time",
            show_forecast=False,
            show_history=True,
            history_rgb="rgb(0,255,0)",
            limit_date_range=False
        )


        fig_pressure = create_time_figure(
            df,
            "pressure.press",
            "Pressure (hPa)",
            show_forecast=True,
            show_history=True,
            history_rgb="rgb(255,51,153)",
            forecast_rgb=FORECAST_RGB,
        )

        fig_humidity = create_time_figure(
            df,
            "humidity",
            "Humidity (%)",
            show_forecast=True,
            show_history=True,
            history_rgb='rgb(255,255,153)',
            forecast_rgb=FORECAST_RGB,
        )

        fig_visibility = create_time_figure(
            df,
            "visibility_distance",
            "Visibility distance (m)",
            show_forecast=True,
            show_history=True,
            history_rgb='rgb(153,0,76)',
            forecast_rgb=FORECAST_RGB,
        )

        # fig_cloud_cover = create_time_figure(
        #     df,
        #     "clouds",
        #     "Cloud coverage (%)",
        #     show_forecast=True,
        #     show_history=True,
        #     history_rgb='rgb(0,255,255)',
        #     forecast_rgb=FORECAST_RGB
        # )

        fig_winddeg = wind_direction_graph(
            df,
            "rgb(204,0,102)",
            forecast_rgb=FORECAST_RGB
        )

        fig_feels_like = create_time_figure(
            df,
            "temperature.feels_like",
            "Temperature feels like (F)",
            show_forecast=True,
            show_history=True,
            column_error_min="temperature.temp_min",
            column_error_max="temperature.temp_max",
            history_rgb="rgb(255,255,0)",
            forecast_rgb=FORECAST_RGB
        )

        graphs_sunrise = create_time_figure(
            df,
            "sunrise.timeofday",
            None,
            show_history=True,
            show_forecast=False,
            history_rgb="rgb(255,255,51)",
            as_type="graphs",
            line_only_graph_type="lines",
            custom_main_trace_label="Sunrise"
        )

        graphs_sunset = create_time_figure(
            df,
            "sunset.timeofday",
            None,
            show_history=True,
            show_forecast=False,
            history_rgb="rgb(255,128,0)",
            as_type="graphs",
            line_only_graph_type="lines",
            custom_main_trace_label="Sunset"
        )

        fig_suntimes = go.Figure(graphs_sunset + graphs_sunrise)
        fig_suntimes.update_layout(
            template="plotly_dark",
            title="Sunrise and Sunsets"
        )

        fig_rainaccu = create_time_figure(
            df,
            "rain.3h",
            "Rain accumulation over past 3h (mm)",
            show_history=True,
            show_forecast=True,
            history_rgb="rgb(7,130,255)",
            forecast_rgb=FORECAST_RGB
        )

        fig_aqi = create_time_figure(
            adf,
            "aqi",
            "Adjusted AQI",
            show_history=True,
            show_forecast=True,
            history_rgb="rgb(7,130,255)",
            forecast_rgb=FORECAST_RGB
        )
        fig_aqi_historical = create_time_figure(
            adf,
            "aqi",
            "Adjusted AQI - All time",
            show_history=True,
            show_forecast=False,
            history_rgb="rgb(7,130,255)",
            limit_date_range=False
        )
        
        for fig in (fig_aqi, fig_aqi_historical):
            fig.update_yaxes(type="category", categoryorder="array", categoryarray=["great", "fair", "moderate", "poor", "hazardous"])

        trace_pollutants_forecasts = False
        trace_pollutants_markers = "lines"

        graphs_co = create_time_figure(
            adf,
            "co",
            None,
            show_history=True,
            show_forecast=trace_pollutants_forecasts,
            history_rgb="rgb(122,120,227)",
            as_type="graphs",
            line_only_graph_type=trace_pollutants_markers,
            custom_main_trace_label="CO"
        )

        graphs_no = create_time_figure(
            adf,
            "no",
            None,
            show_history=True,
            show_forecast=trace_pollutants_forecasts,
            history_rgb="rgb(112,224,208)",
            as_type="graphs",
            line_only_graph_type=trace_pollutants_markers,
            custom_main_trace_label="NO"
        )

        graphs_no2 = create_time_figure(
            adf,
            "no2",
            None,
            show_history=True,
            show_forecast=trace_pollutants_forecasts,
            history_rgb="rgb(224,112,118)",
            as_type="graphs",
            line_only_graph_type=trace_pollutants_markers,
            custom_main_trace_label="NO2"
        )

        graphs_o3 = create_time_figure(
            adf,
            "o3",
            None,
            show_history=True,
            show_forecast=trace_pollutants_forecasts,
            history_rgb="rgb(76,153,60)",
            as_type="graphs",
            line_only_graph_type=trace_pollutants_markers,
            custom_main_trace_label="O3"
        )

        graphs_nh3 = create_time_figure(
            adf,
            "nh3",
            None,
            show_history=True,
            show_forecast=trace_pollutants_forecasts,
            history_rgb="rgb(136,163,60)",
            as_type="graphs",
            line_only_graph_type=trace_pollutants_markers,
            custom_main_trace_label="NH3"
        )

        fig_pollutants = create_time_figure(
            adf,
            "so2",
            None,
            show_history=True,
            show_forecast=trace_pollutants_forecasts,
            history_rgb="rgb(145,145,145)",
            as_type="figure",
            line_only_graph_type=trace_pollutants_markers,
            custom_main_trace_label="SO2"
        )

        for g in [graphs_co, graphs_no, graphs_no2, graphs_o3, graphs_nh3]:
            fig_pollutants.add_trace(g[0])

        fig_pollutants.update_layout(
            template="plotly_dark",
            title="Trace Air Pollutants (ug/m^3)"
        )
        fig_pollutants.update_yaxes(type="log", title="Pollutant concentration")
        

        fig_pm25 = create_time_figure(
            adf,
            "pm2_5",
            "Fine Particulate Matter PM2.5 (ug/m^3)",
            show_history=True,
            show_forecast=True,
            history_rgb="rgb(255,0,0)",
            forecast_rgb=FORECAST_RGB
        )
        fig_pm25.update_yaxes(type="log")

        fig_pm10 = create_time_figure(
            adf,
            "pm10",
            "Coarse Particulate Matter PM10 (ug/m^3)",
            show_history=True,
            show_forecast=True,
            history_rgb="rgb(43,0,237)",
            forecast_rgb=FORECAST_RGB
        )
        fig_pm10.update_yaxes(type="log")

        graphs = [
            dcc.Graph(id='graph_temperature',figure=fig_temp),
            dcc.Graph(id="graph_feels_like", figure=fig_feels_like),
            dcc.Graph(id='graph_precipitation_probability', figure=fig_precipitation_prob),
            dcc.Graph(id='graph_rain_accumulation', figure=fig_rainaccu),
            dcc.Graph(id='graph_windspeed', figure=fig_windspeed),
            dcc.Graph(id="graph_winddeg", figure=fig_winddeg),
            dcc.Graph(id="graph_pressure", figure=fig_pressure),
            dcc.Graph(id="graph_humidty", figure=fig_humidity),
            dcc.Graph(id="graph_visibility", figure=fig_visibility),
            # the cloud coverage data is somewhat too noisy to use for anything useful
            # dcc.Graph(id="graph_cloud_coverage", figure=fig_cloud_cover),
            dcc.Graph(id="graph_suntimes", figure=fig_suntimes),
            dcc.Graph(id="graph_aqi", figure=fig_aqi),
            dcc.Graph(id="graph_pollutants", figure=fig_pollutants),
            dcc.Graph(id="graph_pm25", figure=fig_pm25),
            dcc.Graph(id="graph_pm10", figure=fig_pm10)
        ]
        divs = []


        for i in range(0, len(graphs) - 1, 2):
            d1 = html.Div(graphs[i], className="six columns")
            d2 = html.Div(graphs[i+1], className="six columns")
            divs.append(html.Div([d1, d2], className="row"))

        if len(graphs) % 2 != 0:
            divs.append(html.Div(graphs[-1], className="row"))

        
        graphs_historical = [
            dcc.Graph(id="graph_temp_historical", figure=fig_temp_historical),
            dcc.Graph(id="graph_windspeed_historical", figure=fig_windspeed_historical),
            dcc.Graph(id="graph_aqi_historical", figure=fig_aqi_historical),
        ]
        divs_historical = [html.Div(g) for g in graphs_historical]


        # top rows
        common_style = {"width": "95%", "height": "700px", "margin": "auto", "border": "30px"}
        widget = html.Iframe(
            style=common_style,
            src=WINDY_EMBED
        )


        df["timedeltas_datetime"] = df["datetime"] - datetime.datetime.now()
        df["timedeltas_fetched"] = datetime.datetime.now() - df["fetched_at"]

        # get most recent datum according to timedelta fetched time
        most_recent_historical_idx = df[df["origin"] == "history"]["timedeltas_fetched"].idxmin()
        most_recent_weather = df.loc[most_recent_historical_idx]

        # get most recent forecast for next N days
        most_recent_forecast_idx = df[df["origin"] == "forecast"]["timedeltas_fetched"].idxmin()
        most_recent_forecast_fetched_at = df["fetched_at_str"].loc[most_recent_forecast_idx]

        df_forecast_most_recent = df[
            (df["origin"] == "forecast") & (df["fetched_at_str"] == most_recent_forecast_fetched_at)]


        # df_hm = df_forecast_most_recent.append(most_recent_weather)
        df_hm = df_forecast_most_recent
        df_hm = df_hm.sort_values(by=["datetime"])
        DOWS = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday"
        }
        df_hm["dow"] = df_hm["datetime"].apply(lambda x: DOWS[x.weekday()])
        df_hm["hour"] = df_hm["datetime"].apply(lambda x: x.hour)

        square_index = list(reversed(df_hm["dow"].unique().tolist()))
        hours_index = [0, 3, 6, 9, 12, 15, 18, 21]

        df_square_statuses = pd.DataFrame("no status", columns=hours_index, index=square_index)
        df_square_ints = pd.DataFrame(np.nan, columns=hours_index, index=square_index)

        for entry in df_hm.iterrows():
            d = entry[1]["dow"]
            h = entry[1]["hour"]
            s = entry[1]["status"] # simple status
            df_square_statuses.at[d, h] = "" if s == "no status" else s
            df_square_ints.at[d, h] = OWM_WEATHER_SIMPLE_STATUS_TO_INT[s.lower()]


        # heatmap = go.Heatmap(
        #     z=df_square_ints.values,
        #     y=df_square_ints.index,
        #     x=df_square_ints.columns,
        #     hoverongaps=False,
        #     customdata=df_square_statuses.values,
        #     hovertemplate="forecast:%{customdata}",
        #     colorbar=False
        # )
        #
        # figh = go.Figure(data=heatmap)

        print("df_square_ints")
        print(df_square_ints)


        figh = ff.create_annotated_heatmap(
            df_square_ints.values,
            y=df_square_ints.index.tolist(),
            x=df_square_ints.columns.tolist(),
            annotation_text=df_square_statuses.values,
            colorscale="Bluered",
            showscale=False,
            customdata=df_square_statuses.values,
            hovertemplate="forecast:%{customdata}",
            name="Forecast",
            font_colors=["white", "white"],
        )
        figh.update_layout(
            template="plotly_dark",
            title="Abbreviated forecasts",
            xaxis=dict(title="Hour", showgrid=False),
            yaxis=dict(title="Weekday", showgrid=False)

        )
        g = dcc.Graph(id="graph_heatmap", figure=figh)


        white_text = {"color": "white"}
        wind_speed = int(most_recent_weather["wind.speed"]) if not np.isnan(most_recent_weather["wind.speed"]) else "Unknown"
        wind_cardinal = most_recent_weather["wind.cardinal"]

        right_now = html.Div([
            html.H5(f'Current weather: {most_recent_weather["detailed_status"].capitalize() if most_recent_weather["detailed_status"] else "No current weather"}', style=white_text),
            html.Table([
                html.Tr([
                    html.Td("Temperature", style=white_text),
                    html.Td(f'{int(most_recent_weather["temperature.temp"])}F (feels like {int(most_recent_weather["temperature.feels_like"])}F)', style=white_text)
                ]),
                html.Tr([
                    html.Td("Precipitation prob./accum (3h)", style=white_text),
                    html.Td(f'{most_recent_weather["precipitation_probability"]}/{most_recent_weather["rain.3h"]}mm', style=white_text)
                    # html.Td("K", style=white_text)
                ]),
                html.Tr([
                    html.Td("Wind speed/direction", style=white_text),
                    html.Td(f'{wind_speed} mph @ {wind_cardinal}', style=white_text)
                ]),
                html.Tr([
                    html.Td("Fetched at", style=white_text),
                    html.Td(f'{most_recent_weather["fetched_at_str"]}', style=white_text)
                ])

            ])

        ], style={'padding-left': '40px', 'padding-right': '40px', "marginLeft": "auto", "marginRight": "auto"})
        col1 = html.Div([right_now, g], className="six columns")
        col2 = html.Div(widget, className="six columns")
        toprow = html.Div([col1, col2], className="row")
        return html.Div([toprow] + [html.H3("Recent Weather Metrics", style=white_text)] + divs + [html.H3("Full History", style=white_text)] + divs_historical)
    except BaseException:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tbf = traceback.format_exception(exc_type, exc_value, exc_traceback)
        logging.critical("Critical wdash webapp error:")
        logging.critical(tbf)


app.layout = html.Div(
    children=[
        dcc.Interval("interval", interval=INTERVAL * 1000),
        html.H1(children='Weather dashboard for Truckee, CA', style={"color": "white"}),
        html.Div(children='''
            Updated with data from the OpenWeatherMaps API, pyOWM, and Windy.
        ''', style={"color": "white"}),
        html.Br(),
        html.Br(),
        dcc.Loading(children=html.Div(id="all_info"), type="dot")

    ],
    style={"backgroundColor": "rgb(17,17,17)"}
)

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    #app.run_server(debug=True, port=SERVER_PORT)
    app.run_server(debug=False, host="0.0.0.0", port=SERVER_PORT)


    # df = create_aqi_df()
    # print(df)
