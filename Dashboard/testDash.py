# Importamos las librerias mínimas necesarias
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import Input, Output, dcc, html, State
import dash_bootstrap_components as dbc
import plotly.express as px
import logging

# Leemos las bases de datos
df_airports = pd.read_parquet("EDA/df_airports.parquet")
# df_flights = pd.read_parquet("../EDA/df_flights.parquet")
# df_airports_date = pd.read_parquet("../EDA/df_airports_date.parquet")

# Pie Chart
delay_labels = ["AIR_SYSTEM_DELAY","SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","OTHER_DELAY"]
i = 0
values1 = df_airports[delay_labels].iloc[i]
fig_pie = go.Figure()
fig_pie.add_trace(go.Pie(labels=delay_labels, values=values1, direction ='clockwise', marker_colors=px.colors.qualitative.Vivid, hole=.3))
fig_pie.update_layout(title_text="Average Delay Distribution by Airport", legend_title="Delay Cause", template="plotly_dark",
                    legend=dict(orientation="h", y=-0.02, x =0.08))

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H2('AIRPORTS ANALYSIS', className='text-center text-primary, mb-3'))),  # header row
        
        dbc.Row([  # start of second row
            dbc.Col([  # first column on second row
            html.H5('Map', className='text-center'),
            dcc.Graph(id='map-main',
                      style={'height':550}),
            html.Hr(),
            ], width={'size': 8, 'offset': 0, 'order': 1}),  # width first column on second row
            dbc.Col([  # second column on third row
                html.H5('Distribution', className='text-center'),
                dcc.Graph(id='pie-main',
                      figure = fig_pie,
                      style={'height':550}),
            ], width={'size': 4, 'offset': 0, 'order': 2}),  # width second column on second row
        ]),  # end of second row      
    ], fluid=True)

@app.callback(
    Output('map-main', 'figure'))
def create_graph():
    dff = df_airports
    fig = px.scatter_geo(dff, lat="ORIGIN_LATITUDE", lon = "ORIGIN_LONGITUDE",
                     size= "FLIGHTS", # size of markers
                     size_max= 30,
                     color= "DELAYED_PERCENTAGE", # which column to use to set the color of markers
                     scope="usa",
                     text = "ORIGIN_AIRPORT",
                     hover_data  = ["ORIGIN_CITY"],
                     color_continuous_scale='RdYlGn_r',
                     template="plotly_dark")
    fig.update_traces(textposition="top center")
    fig.update_layout(
        title="Origin airports with number of departing flights and percentage of delayed flights <br><br><sup>Size indicates the number of departing flights</sup>",
        legend_title="Causa del Retraso",
    )
    return fig

if __name__ == '__main__':
    app.run_server()