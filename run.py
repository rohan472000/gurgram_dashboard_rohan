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

def idw(sensor_df, grid_gdf, value_col='value', power=2):
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

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Daily Interpolated Air Quality - Gurugram", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Select Date:"),
        dcc.Slider(
            id="date-slider",
            min=0,
            max=len(dates) - 1,
            step=1,
            marks={i: dates[i].strftime('%d-%b') for i in range(0, len(dates), max(1, len(dates) // 10))},
            value=0
        ),
        html.Label("Select Variable:"),
        dcc.Dropdown(
            id="parameter-dropdown",
            options=[{'label': col, 'value': col} for col in ['PM2.5', 'Temperature', 'Humidity']],
            value='PM2.5',
            clearable=False
        )
    ], style={'padding': '10px 50px'}),
    dcc.Graph(id="map-plot")
])

@app.callback(
    Output("map-plot", "figure"),
    Input("date-slider", "value"),
    Input("parameter-dropdown", "value")
)
def update_map(slider_index, selected_param):
    date = dates[slider_index]
    df = daily_obs_df[daily_obs_df['Timestamp'] == date].dropna(subset=[selected_param, 'Latitude', 'Longitude'])
    if df.empty:
        return px.scatter_mapbox()
    df = df.rename(columns={selected_param: 'value'})
    interpolated = idw(df, clipped_grid, 'value')
    interpolated = interpolated.dropna(subset=['value'])

    vmin, vmax = {
        "PM2.5": (0, 500),
        "Temperature": (0, 50),
        "Humidity": (0, 100)
    }[selected_param]

    fig = px.choropleth_mapbox(
        interpolated,
        geojson=json.loads(interpolated.to_json()),
        locations=interpolated.index,
        color="value",
        color_continuous_scale="YlOrRd",
        range_color=(vmin, vmax),
        mapbox_style="carto-positron",
        center={"lat": 28.45, "lon": 77.02},
        zoom=10.5,
        opacity=0.6,
        height=700
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title=f"{selected_param} on {date.strftime('%d-%b-%Y')}")
    return fig

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)

