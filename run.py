import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import json
import glob
from shapely.geometry import box
from scipy.spatial import cKDTree
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

boundary = gpd.read_file("GMDA - Boundary Georef/extracted_kml/doc.kml", driver="KML").to_crs(epsg=4326)
xmin, ymin, xmax, ymax = boundary.total_bounds
grid_size = 0.009
cols = np.arange(xmin, xmax, grid_size)
rows = np.arange(ymin, ymax, grid_size)
grid_cells = [box(x, y, x + grid_size, y + grid_size) for x in cols for y in rows]
grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=boundary.crs)
clipped_grid = gpd.overlay(grid, boundary, how='intersection')
clipped_grid['zone_id'] = range(1, len(clipped_grid) + 1)

file_paths = glob.glob("filtered_master_file_api/*.csv")
df_list = []
for file_path in file_paths:
    parts = file_path.split("_")
    if len(parts) < 7:
        continue
    sensor_name = parts[4]
    thing_id = parts[6]
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df['Sensor_Name'] = sensor_name
    df['Thing_ID'] = thing_id
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce').round(4)
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce').round(4)
    for col in ['PM2.5', 'Temperature', 'Humidity']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['PM2.5', 'Temperature', 'Humidity', 'Latitude', 'Longitude'])
    df_numeric = df.set_index('Timestamp')[['PM2.5', 'Temperature', 'Humidity']].resample('D').mean().reset_index()
    df_meta = df.set_index('Timestamp')[['Sensor_Name', 'Thing_ID', 'Latitude', 'Longitude']].resample('D').first().reset_index()
    df_combined = pd.merge(df_numeric, df_meta, on='Timestamp', how='left')
    df_combined['Sensor_Location'] = df_combined['Sensor_Name'] + '_' + df_combined['Thing_ID']
    df_list.append(df_combined)

all_df = pd.concat(df_list, ignore_index=True)
all_df = all_df[all_df['Timestamp'] > '2024-08-31']
daily_obs_df = all_df.groupby(['Timestamp', 'Sensor_Location', 'Latitude', 'Longitude']).agg({
    'PM2.5': 'mean',
    'Temperature': 'mean',
    'Humidity': 'mean'
}).reset_index()
dates = sorted(daily_obs_df['Timestamp'].unique())

def iidw(sensor_df, grid_gdf, value_col='value', power=2):
    points = sensor_df[['Longitude', 'Latitude']].values
    values = sensor_df[value_col].values
    grid_coords = np.array([geom.centroid.coords[0] for geom in grid_gdf.geometry])
    tree = cKDTree(points)
    dist, idx = tree.query(grid_coords, k=min(5, len(points)), distance_upper_bound=np.inf)
    weights = 1 / (dist ** power + 1e-6)
    weights = weights / weights.sum(axis=1)[:, None]
    interpolated = np.sum(weights * values[idx], axis=1)
    grid_gdf = grid_gdf.copy()
    grid_gdf[value_col] = interpolated
    return grid_gdf
    
def idw(sensor_df, grid_gdf, value_col='value', power=2):
    points = sensor_df[['Longitude', 'Latitude']].values
    values = sensor_df[value_col].values
    grid_coords = np.array([geom.centroid.coords[0] for geom in grid_gdf.geometry])
    
    k = min(5, len(points))
    tree = cKDTree(points)
    dist, idx = tree.query(grid_coords, k=k, distance_upper_bound=np.inf)

    if k == 1:
        dist = dist[:, np.newaxis]
        idx = idx[:, np.newaxis]

    mask = np.isinf(dist)
    dist[mask] = 1e-6  
    weights = 1 / (dist ** power + 1e-6)
    weights[mask] = 0  
    weights_sum = weights.sum(axis=1)[:, None]
    weights = np.divide(weights, weights_sum, out=np.zeros_like(weights), where=weights_sum != 0)

    interpolated = np.sum(weights * values[idx], axis=1)

    grid_gdf = grid_gdf.copy()
    grid_gdf[value_col] = interpolated
    return grid_gdf

def generate_lstm_predictions(df, forecast_days=7):
    predictions_all = []
    for (lat, lon), group_df in df.groupby(['Latitude', 'Longitude']):
        group_df = group_df.sort_values('Timestamp')
        features = ['PM2.5', 'Temperature', 'Humidity']

        if group_df[features].isnull().any().any() or len(group_df) < 15:
            continue

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(group_df[features])
        X, y = [], []
        seq_len = 7
        for i in range(len(scaled_data) - seq_len):
            X.append(scaled_data[i:i + seq_len])
            y.append(scaled_data[i + seq_len][0])

        X, y = np.array(X), np.array(y)
        if len(X) < 5:
            continue

        model = Sequential([
            LSTM(32, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=5, verbose=2)

        input_seq = scaled_data[-seq_len:]
        future_preds = []
        for _ in range(forecast_days):
            pred = model.predict(input_seq.reshape(1, seq_len, 3), verbose=0)[0][0]
            future_preds.append(pred)
            next_input = np.array([pred, input_seq[-1][1], input_seq[-1][2]])
            input_seq = np.vstack([input_seq[1:], next_input])

        restored_preds = scaler.inverse_transform(
            np.column_stack([future_preds,
                             [input_seq[-1][1]] * forecast_days,
                             [input_seq[-1][2]] * forecast_days])
        )[:, 0]

        # dates = pd.date_range(start=group_df['Timestamp'].max() + pd.Timedelta(days=1), periods=forecast_days)
        forecast_start = df['Timestamp'].max() + pd.Timedelta(days=1)
        dates = pd.date_range(start=forecast_start, periods=forecast_days)

        pred_df = pd.DataFrame({
            'Latitude': lat,
            'Longitude': lon,
            'Temperature': input_seq[-1][1],
            'Humidity': input_seq[-1][2],
            'value': restored_preds,
            'Timestamp': dates
        })
        predictions_all.append(pred_df)

    return pd.concat(predictions_all, ignore_index=True)

future_df = generate_lstm_predictions(daily_obs_df)
predicted_dates = sorted(future_df['Timestamp'].unique())

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Daily Interpolated Air Quality - Gurugram", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select Variable:"),
        dcc.Dropdown(
            id="parameter-dropdown",
            options=[
                {'label': 'PM2.5 (µg/m³)', 'value': 'PM2.5'},
                {'label': 'Temperature (°C)', 'value': 'Temperature'},
                {'label': 'Humidity (%)', 'value': 'Humidity'},
                {'label': 'Predicted PM2.5 (µg/m³)', 'value': 'Predicted PM2.5'}
            ],
            value='PM2.5',
            clearable=False
        ),
        html.Br(),

        html.Div(id="slider-container", children=[
            html.Div(
                dcc.Slider(
                    id="date-slider-historical",
                    min=0,
                    max=len(dates) - 1,
                    step=1,
                    marks={i: dates[i].strftime('%d-%b') for i in range(0, len(dates), max(1, len(dates)//10))},
                    value=0
                ),
                id="historical-slider-wrapper"
            ),
            html.Div(
                # dcc.Slider(
                #     id="date-slider-predicted",
                #     min=0,
                #     max=len(predicted_dates) - 1,
                #     step=1,
                #     marks={i: predicted_dates[i].strftime('%d-%b') for i in range(len(predicted_dates))},
                #     value=0
                # )
                dcc.Slider(
                    id="date-slider-predicted",
                    min=0,
                    max=len(predicted_dates) - 1,
                    step=1,
                    marks={i: predicted_dates[i].strftime('%d-%b') for i in range(0, len(predicted_dates), max(1, len(predicted_dates)//7))},
                    value=0
                )
                ,
                id="predicted-slider-wrapper",
                style={"display": "none"}
            )
        ])
    ], style={'padding': '10px 50px'}),

    dcc.Graph(id="map-plot")
])

@app.callback(
    Output("historical-slider-wrapper", "style"),
    Output("predicted-slider-wrapper", "style"),
    Input("parameter-dropdown", "value")
)
def toggle_slider(selected_param):
    if selected_param == "Predicted PM2.5":
        return {"display": "none"}, {"display": "block"}
    else:
        return {"display": "block"}, {"display": "none"}

@app.callback(
    Output("map-plot", "figure"),
    Input("parameter-dropdown", "value"),
    Input("date-slider-historical", "value"),
    Input("date-slider-predicted", "value")
)
def update_map(selected_param, hist_index, pred_index):
    if selected_param == 'Predicted PM2.5':
        date = predicted_dates[pred_index]
        df = future_df[future_df['Timestamp'] == date][['Latitude', 'Longitude', 'value']]
    else:
        date = dates[hist_index]
        df = daily_obs_df[daily_obs_df['Timestamp'] == date].dropna(subset=[selected_param, 'Latitude', 'Longitude'])
        df = df.rename(columns={selected_param: 'value'})
        df = df[['Latitude', 'Longitude', 'value']]

    if df.empty:
        return px.scatter_mapbox(
            lat=[], lon=[], zoom=10.5, mapbox_style="carto-positron",
            center={"lat": 28.45, "lon": 77.02}, height=700
        ).update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title="No data available for selected date")

    grid_with_preds = idw(df, clipped_grid, 'value')

    vmin, vmax = {
        "PM2.5": (0, 500),
        "Temperature": (0, 50),
        "Humidity": (0, 100),
        "Predicted PM2.5": (0, 500)
    }[selected_param]

    unit = {
        "PM2.5": "µg/m³",
        "Temperature": "°C",
        "Humidity": "%",
        "Predicted PM2.5": "µg/m³"
    }[selected_param]

    fig = px.choropleth_mapbox(
        grid_with_preds,
        geojson=json.loads(grid_with_preds.to_json()),
        locations=grid_with_preds.index,
        color="value",
        color_continuous_scale="YlOrRd",
        range_color=(vmin, vmax),
        mapbox_style="carto-positron",
        center={"lat": 28.45, "lon": 77.02},
        zoom=10.5,
        opacity=0.6,
        height=700
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title=f"{selected_param} on {date.strftime('%d-%b-%Y')} ({unit})")
    fig.update_coloraxes(colorbar_title=unit)
    return fig

if __name__ == "__main__":
    is_render = os.environ.get("RENDER", None)
    port = int(os.environ.get("PORT", 8050))
    host = "0.0.0.0" if is_render else "127.0.0.1"
    app.run(host=host, port=port, debug=True)
