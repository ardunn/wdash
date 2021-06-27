# wdash
a weather dashboard built from OpenWeatherMaps APi, MongoDB, Windy, and Plotly Dash


# Picture Demo
![demo](demo.png)

# Video Demo
![demo](demo.gif)


# Configure

All the variables needed to configure `wdash` are in `config.yml`.

## `CITY_ID`
The city ID (an integer) for your city as per [OpenWeatherMaps search API](https://openweathermap.org/find). Enter your city and country name and the resulting url (e.g., https://openweathermap.org/city/5403676) contains the integer city ID (in this example, 5403676).


## `DB_HOST`
The host of your mongodb instance, e.g., `localhost`

## `DB_PORT`
The port of your mongodb instance, e.g., 27017

## `DB_NAME`
The name of the mongodb database you'd like to use or create to hold your wdash data, e.g., `wdash`

## `INTERVAL`
The inteval to update (gather current weathers and forecasts), in seconds. The default `1800` corresponds to  every 30 minutes.

## `PORT`
The port you'd like the `wdash` interface to run on.

## `API_KEY`
Your OpenWeatherMaps API key. Only the free-tier API key is required for full functionality.

## `WINDY_EMBED`
The embeddable url for your windy map, which you can obtain from [Windy](https://www.windy.com/-Embed-widget-on-page/widgets?39.339,-120.173,5)


# Running `wdash`

Simply run


```
source run.sh
```

In the `wdash` directory for the daemon and app to run in the background. All output will be logged to `log.txt` in the `wdash` directory.

