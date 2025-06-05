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

# === Load Boundary and Create Grid ===
boundary = gpd.read_file("GMDA - Boundary Georef/extracted_kml/doc.kml", driver="KML").to_crs(epsg=4326)
xmin, ymin, xmax, ymax = boundary.total_bounds
grid_size = 0.009
cols = np.arange(xmin, xmax, grid_size)
rows = np.arange(ymin, ymax, grid_size)
grid_cells = [box(x, y, x + grid_size, y + grid_size) for x in cols for y in rows]
grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=boundary.crs)
clipped_grid = gpd.overlay(grid, boundary, how='intersection')
clipped_grid['zone_id'] = range(1, len(clipped_grid) + 1)

# === Load and Process Sensor Data ===
allowed_qrcodes = {'5WZHYW9W','ELLUJP9P','FVPFR6N8','HN1ZDFQ4','HQ9YMO1D','IR3BO10Z','LBMZO9OW','N3C5VR4U','NB8UUJGI','ORTZ0IBL','Q4XCUI8G','V3O3XQX7','Z4164TFS','ELLUJP9P'}  
file_paths = glob.glob("filtered_master_file_api/*.csv")
df_list = []
for file_path in file_paths:
    
    parts = file_path.split("_")
    if len(parts) < 7:
        continue
        
    # qrcode = parts[4].replace(".csv", "")  
    # if qrcode not in allowed_qrcodes:
    #     continue    
        
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

# === Prediction Stub ===
predicted_dates = pd.date_range(start='2025-06-01', periods=7)
future_df = pd.DataFrame({
    'Latitude': np.random.uniform(28.35, 28.50, 100),
    'Longitude': np.random.uniform(76.90, 77.10, 100),
    'value': np.random.uniform(5, 300, 100),
    'Timestamp': np.random.choice(predicted_dates, 100)
})

# === IDW Interpolation ===
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

# === Dash App Setup ===
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
            html.Div(dcc.Slider(
                id="date-slider-historical",
                min=0,
                max=len(dates) - 1,
                step=1,
                marks={i: dates[i].strftime('%d-%b') for i in range(0, len(dates), max(1, len(dates)//10))},
                value=0), id="historical-slider-wrapper", style={"display": "block"}),
            html.Div(dcc.Slider(
                id="date-slider-predicted",
                min=0,
                max=len(predicted_dates) - 1,
                step=1,
                marks={i: predicted_dates[i].strftime('%d-%b') for i in range(0, len(predicted_dates), max(1, len(predicted_dates)//7))},
                value=0), id="predicted-slider-wrapper", style={"display": "none"})
        ])
    ], style={'padding': '10px 50px'}),
    dcc.Graph(id="map-plot")
])

# === Toggle Sliders ===
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

# === Update Map Callback ===
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
        return px.scatter_mapbox(lat=[], lon=[], zoom=10.5, mapbox_style="carto-positron",
                                 center={"lat": 28.45, "lon": 77.02}, height=700)

    grid_with_preds = idw(df, clipped_grid, 'value')
    unit = {"PM2.5": "µg/m³", "Temperature": "°C", "Humidity": "%", "Predicted PM2.5": "µg/m³"}[selected_param]

    if selected_param in ["PM2.5", "Predicted PM2.5"]:
        bins = [0, 12, 35.4, 55.4, 150.4, 250.4, 500]
        range_labels = ['0–12', '12.1–35.4', '35.5–55.4', '55.5–150.4', '150.5–250.4', '250.5+']
        label_to_color = {
            '0–12': 'green',
            '12.1–35.4': 'yellow',
            '35.5–55.4': 'orange',
            '55.5–150.4': 'red',
            '150.5–250.4': 'purple',
            '250.5+': 'maroon'
        }

        grid_with_preds['Category'] = pd.cut(grid_with_preds['value'], bins=bins, labels=range_labels, include_lowest=True)
        grid_with_preds['hover_text'] = 'Value: ' + grid_with_preds['value'].round(1).astype(str) + ' ' + unit + '<br>Bin: ' + grid_with_preds['Category'].astype(str)

        # Ensure all categories appear in legend
        dummy_df = pd.DataFrame({
            'geometry': [None]*len(range_labels),
            'Category': range_labels,
            'hover_text': ['']*len(range_labels),
            'value': [0]*len(range_labels)
        })
        full_df = pd.concat([grid_with_preds, dummy_df], ignore_index=True)

        fig = px.choropleth_mapbox(
            full_df,
            geojson=json.loads(clipped_grid.to_json()),
            locations=full_df.index,
            color="Category",
            color_discrete_map=label_to_color,
            hover_name="hover_text",
            mapbox_style="carto-positron",
            center={"lat": 28.45, "lon": 77.02},
            zoom=10.5,
            opacity=0.6,
            height=700,
            category_orders={"Category": range_labels}
        )
        fig.update_layout(title=f"{selected_param} on {date.strftime('%d-%b-%Y')} ({unit})",
                          legend_title="PM2.5 Bins (µg/m³)",
                          legend=dict(itemsizing='constant'))
    else:
        vmin, vmax = {"Temperature": (0, 50), "Humidity": (0, 100)}[selected_param]
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
        fig.update_coloraxes(colorbar_title=unit)
        fig.update_layout(title=f"{selected_param} on {date.strftime('%d-%b-%Y')} ({unit})")

    return fig

if __name__ == "__main__":
    is_render = os.environ.get("RENDER", None)
    port = int(os.environ.get("PORT", 8050))
    host = "0.0.0.0" if is_render else "127.0.0.1"
    app.run(host=host, port=port, debug=True)
