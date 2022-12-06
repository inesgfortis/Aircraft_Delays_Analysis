# Importamos las librerias mínimas necesarias
import pandas as pd
# import numpy as np
# import pickle
# import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import Input, Output, html, callback
#from datetime import date,datetime, timedelta

import dash_bootstrap_components as dbc
from dash import dcc
import plotly.express as px

## DATA
df = pd.read_parquet("Preprocessing/flightsFilteredCleaned.parquet")

# Needed constants
delay_labels = ["AIR_SYSTEM_DELAY","SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","OTHER_DELAY"]
df_delayed = df[df["ARRIVAL_DELAY"]>0]
df_delayed['MAIN_DELAY_CAUSE'] = df_delayed[delay_labels].idxmax(axis=1)
variables_to_group_by = ["ORIGIN_AIRPORT","ORIGIN_AIRPORT_NAME","ORIGIN_CITY","ORIGIN_STATE"]
variables_to_group_by2 = ["DESTINATION_CITY","DESTINATION_STATE","ORIGIN_AIRPORT"]
variables_strictly_needed2 = ["AIR_SYSTEM_DELAY","SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","OTHER_DELAY",
                                "ARRIVAL_DELAY", "ORIGIN_AIRPORT"]
variables_strictly_needed = ["ORIGIN_AIRPORT","ORIGIN_AIRPORT_NAME","ORIGIN_CITY","ORIGIN_STATE","ARRIVAL_DELAY","ORIGIN_LATITUDE","ORIGIN_LONGITUDE"]
MONTH = {1: 'Janauary', 2:'February',3:'March',4:'April',5:'May',6:'June',
		 7:'July',8:'August',9:'September',10:'October',11:'November', 12:'December'}



## Dash
dash.register_page(__name__, path = "/", name = "Airports")


########################################################################################################################
# TAB CONTENT
########################################################################################################################

layout = [
        dbc.Row(dbc.Col(html.H2('AIRPORTS ANALYSIS', className='text-center text-primary, mb-3'))),  # header row
        
        dbc.Row([  # start of second row
            dbc.Col([  # first column on second row
                html.H5('Map', className='text-center'),
                dcc.Graph(id='map-main',
                        hoverData={"points": [{"text": "ATL"}]},
                        style={'height':600}),
                html.Hr(),
            ], width={'size': 7, 'offset': 0, 'order': 1}),  # width first column on second row
            dbc.Col([  # second column on third row
                html.H5('Distribution', className='text-center'),
                dcc.Graph(id='pie-main',
                      style={'height':600}),
                html.Hr(),
            ], width={'size': 5, 'offset': 0, 'order': 2}),  # width second column on second row
        ]),  # end of second row 

        dbc.Row([  # start of third row
            dbc.Col([  # first column on third row
                dcc.Graph(id='bar-main',
                      style={'height':500}),
            ], width={'size': 7, 'offset': 0, 'order': 1}),
            dbc.Col([  # second column on third row
                    html.H5('Filters', className='text-center'),
                    dbc.Label("Date"),
                    dcc.RangeSlider(1, 12, 1, marks=MONTH, value=[1,12], id="range-slider-main"),
                    html.Hr(),
                ], width={'size': 5, 'offset': 0, 'order': 2}) 
        ]),  # end of third row  
    ]
 


########################################################################################################################
# CALLBACKS
########################################################################################################################

@callback(
    Output('map-main', 'figure'),
    [Input("range-slider-main", "value")])

def update_map(value):    
    dff = df[df["DATE"].dt.month.isin(list(range(value[0],value[1]+1)))]
    df_map = dff[variables_strictly_needed].groupby(variables_to_group_by).mean()
    df_map["FLIGHTS"] = dff.groupby(variables_to_group_by).size()
    df_map["DELAYED_FLIGHTS"] = dff[dff["ARRIVAL_DELAY"]>0].groupby(variables_to_group_by).size()
    df_map["DELAYED_PERCENTAGE"] = df_map["DELAYED_FLIGHTS"]/df_map["FLIGHTS"]
    df_map = df_map.reset_index() 

    fig_map = px.scatter_geo(df_map, lat="ORIGIN_LATITUDE", lon = "ORIGIN_LONGITUDE",
                        size= "FLIGHTS", # size of markers
                        size_max= 30,
                        color= "DELAYED_PERCENTAGE", # which column to use to set the color of markers
                        scope="usa",
                        text = "ORIGIN_AIRPORT",
                        hover_data  = ["ORIGIN_CITY"],
                        color_continuous_scale='RdYlGn_r',
                        template="plotly_dark")
    fig_map.update_traces(textposition="top center")
    fig_map.update_layout(
        title="Origin airports with number of departing flights and percentage of delayed flights\
                <br><sup>Size indicates the number of departing flights</sup>\
                <sup>Maintain the mouse in an airport to obtain its full information</sup>",
        legend_title="Causa del Retraso")
        
    return fig_map
    
@callback(
    Output('pie-main', 'figure'),
    [Input('map-main', 'hoverData'),
    Input("range-slider-main", "value")])


def update_pie(hoverdata, value):
    # Pie Chart
    airport = hoverdata['points'][0]['text']

    df_subplot1 = df_delayed[df_delayed["DATE"].dt.month.isin(list(range(value[0],value[1]+1)))]
    dff = df[df["DATE"].dt.month.isin(list(range(value[0], value[1]+1)))]
    df_subplot2 = dff[variables_strictly_needed2].groupby("ORIGIN_AIRPORT").mean().round(3)
    df_subplot2["FLIGHTS"] = dff.groupby("ORIGIN_AIRPORT").size()
    df_subplot2["DELAYED_FLIGHTS"] = dff[dff["ARRIVAL_DELAY"]>0].groupby("ORIGIN_AIRPORT").size()
    df_subplot2["DELAYED_PERCENTAGE"] = df_subplot2["DELAYED_FLIGHTS"]/df_subplot2["FLIGHTS"]
    df_subplot2 = df_subplot2.reset_index() 

    fig_pie = make_subplots(rows=1, cols=2, subplot_titles= ["Delayed Flights by Main Cause","Average Delay Distribution"],
                    specs=[[{"type": "pie"}, {"type": "pie"}]], horizontal_spacing = 0.03, vertical_spacing = 0.03)
    #subplot 1
    values1 = df_subplot1[df_subplot1["ORIGIN_AIRPORT"]==airport]["MAIN_DELAY_CAUSE"].value_counts().reindex(delay_labels)
    fig_pie.add_trace(go.Pie(labels=values1.index, values=values1, direction ='clockwise', marker_colors=px.colors.qualitative.Vivid, 
                                hole=.3, title ='{:,}<br>delayed<br>flights'.format(values1.sum()),
                                hoverinfo='label+percent', textinfo='value'), row=1, col=1)
    #subplot 2
    values2 = df_subplot2[delay_labels].iloc[df_subplot2[df_subplot2["ORIGIN_AIRPORT"]==airport].index[0]]
    fig_pie.add_trace(go.Pie(labels=values2.index, values=values2, direction ='clockwise', marker_colors=px.colors.qualitative.Vivid, 
                                hole=.3, title = "%.3f <br> seconds" % (values2.sum()),
                                hoverinfo='label+percent', textinfo='value'), row=1, col=2)
    # layout
    fig_pie.update_layout(title_text="Delayed Flights Analysis in %s" % (airport),
                    legend_title="Delay Cause", template="plotly_dark",
                    legend=dict(orientation="h", y=0, x =-0.04))
    fig_pie.update_annotations(yshift=-10)

    return fig_pie

@callback(
    Output('bar-main', 'figure'),
    [Input('map-main', 'hoverData'),
    Input("range-slider-main", "value")])

def update_bar(hoverdata, value):    
    airport = hoverdata['points'][0]['text']
    df_dest = df[df["ORIGIN_AIRPORT"]==airport].groupby(variables_to_group_by2)[["ARRIVAL_DELAY"]].count()
    df_dest["DELAYED_FLIGHTS"] = df[(df["ARRIVAL_DELAY"]>0) & (df["ORIGIN_AIRPORT"]==airport)].groupby(variables_to_group_by2).size()
    df_dest["DELAYED_PERCENTAGE"] = (df_dest["DELAYED_FLIGHTS"]/df_dest["ARRIVAL_DELAY"]).round(4)
    df_dest = df_dest.sort_values("ARRIVAL_DELAY",ascending=False).reset_index()
    df_dest = df_dest.head(10)

    fig_bar = go.Figure([go.Bar(x=df_dest["DESTINATION_CITY"], y=df_dest["ARRIVAL_DELAY"], name="Total", 
                        marker_color=px.colors.qualitative.Vivid[0])])
    fig_bar.add_bar(x=df_dest["DESTINATION_CITY"], y=df_dest["DELAYED_FLIGHTS"], name="Delayed",
                marker_color=px.colors.qualitative.Vivid[1])
    fig_bar.update_layout(title_text="Top 10 most frequent destinations from %s" % (airport), legend_title="Number of flights", template="plotly_dark", 
                    barmode='overlay', hovermode="x unified", legend=dict(x=1, y=1.02, xanchor="right", yanchor="bottom", orientation="h"))
    return fig_bar


########################################################################################################################
########################################################################################################################