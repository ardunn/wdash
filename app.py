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
import pandas as pd
import numpy as np
import logging

handler = logging.StreamHandler()
formatter = logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(THIS_DIR, "config.yml"), "r") as config_file:
    CONFIG = yaml.safe_load(config_file)

HOST = CONFIG["DB_HOST"]
PORT = CONFIG["DB_PORT"]
NAME = CONFIG["DB_NAME"]
INTERVAL = CONFIG["INTERVAL"]

mc = MongoClient(host=HOST, port=PORT)
cw = mc[NAME].weather_current
cf = mc[NAME].weather_forecasted


N_FORECASTS_SHOWN = 10
N_FORECAST_WINDOW = 7
FORECAST_WINDOW_UNITS = "d"
N_WIND_STAR_WINDOW = 5
WIND_STAR_WINDOW_UNITS = "h"
FORECAST_RGB = "rgb(160,160,160)"

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


    # manually fix a few columns
    for c in ["precipitation_probability", "snow.3h", "rain.3h"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
        else:
            logging.info(f"Expected column '{c}' not found in dataframe!")

    return df


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
        showlegend=False
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
        as_type="figure"
):
    df_forecast = df[df["origin"] == "forecast"]
    df_history = df[df["origin"] == "history"]

    graphs = []

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
                "Recorded data"
            )
        else:
            logging.info(f"Creatimg '{column_main}' graph with no error bars")
            history_graphs = get_line_graphs(
                df_history,
                "datetime",
                column_main,
                history_rgb,
                3,
                "lines",
                "Recorded data"
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


# def weather_heatmap



def generate_page():
    df = create_df()

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

    fig_cloud_cover = create_time_figure(
        df,
        "clouds",
        "Cloud coverage (%)",
        show_forecast=True,
        show_history=True,
        history_rgb='rgb(0,255,255)',
        forecast_rgb=FORECAST_RGB
    )

    fig_winddeg = wind_direction_graph(
        df,
        "rgb(204,0,102)",
        forecast_rgb=FORECAST_RGB
    )

    graphs = [
        dcc.Graph(id='graph_temperature',figure=fig_temp),
        dcc.Graph(id='graph_precipitation_probability', figure=fig_precipitation_prob),
        dcc.Graph(id='graph_windspeed', figure=fig_windspeed),
        dcc.Graph(id="graph_winddeg", figure=fig_winddeg),
        dcc.Graph(id="graph_pressure", figure=fig_pressure),
        dcc.Graph(id="graph_humidty", figure=fig_humidity),
        dcc.Graph(id="graph_visibility", figure=fig_visibility),
        dcc.Graph(id="graph_cloud_coverage", figure=fig_cloud_cover)
    ]
    divs = []


    for i in range(0, len(graphs) - 1, 2):
        d1 = html.Div(graphs[i], className="six columns")
        d2 = html.Div(graphs[i+1], className="six columns")
        divs.append(html.Div([d1, d2], className="row"))

    if len(graphs) % 2 != 0:
        divs.append(html.Div(graphs[-1], className="row"))
    return html.Div(divs)



app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        dcc.Interval("interval", interval=INTERVAL * 1000),
        html.H1(children='Weather dashboard for Truckee, CA', style={"color": "white"}),

        html.Div(children='''
            Updated with data from the OpenWeatherMaps API and pyOWM.
        ''', style={"color": "white"}),
        generate_page(),

    ],
    style={"backgroundColor": "rgb(17,17,17)"}
)


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    app.run_server(debug=True)
