# Importamos las librerias mínimas necesarias
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import Input, Output, dcc, html, State
import dash_bootstrap_components as dbc
import plotly.express as px
import logging
from plotly.subplots import make_subplots

# Leemos las bases de datos

df = pd.read_parquet("Preprocessing/flightsFilteredCleaned.parquet")

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
                      figure = fig_map,
                      hoverData={"points": [{"text": "ATL"}]},
                      style={'height':600}),
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
                dcc.Graph(id='filter-main',
                      style={'height':500}),
                html.Hr(),
            ], width={'size': 5, 'offset': 0, 'order': 2}) 
        ]),  # end of third row  
    ], fluid=True)

# callbacks
@app.callback(
    Output('pie-main', 'figure'),
    Input('map-main', 'hoverData'))
def update_pie(hoverdata):
    # Pie Chart
    airport = hoverdata['points'][0]['text']
    fig_pie = make_subplots(rows=1, cols=2, subplot_titles= ["Delayed Flights by Main Cause","Average Delay Distribution"],
                    specs=[[{"type": "pie"}, {"type": "pie"}]], horizontal_spacing = 0.03, vertical_spacing = 0.03)

    #subplot 1
    values1 = df_delayed[df_delayed["ORIGIN_AIRPORT"]==airport]["MAIN_DELAY_CAUSE"].value_counts().reindex(delay_labels)
    fig_pie.add_trace(go.Pie(labels=values1.index, values=values1, direction ='clockwise', marker_colors=px.colors.qualitative.Vivid, 
                                hole=.3, title ='{:,}<br>delayed<br>flights'.format(values1.sum()),
                                hoverinfo='label+percent', textinfo='value'), row=1, col=1)

    #subplot 2
    values2 = df_airports[delay_labels].iloc[df_airports[df_airports["ORIGIN_AIRPORT"]==airport].index[0]]
    fig_pie.add_trace(go.Pie(labels=values2.index, values=values2, direction ='clockwise', marker_colors=px.colors.qualitative.Vivid, 
                                hole=.3, title = "%.3f <br> seconds" % (values2.sum()),
                                hoverinfo='label+percent', textinfo='value'), row=1, col=2)

    fig_pie.update_layout(title_text="Airport: %s" % (airport),
                    legend_title="Delay Cause", template="plotly_dark",
                    legend=dict(orientation="h", y=0, x =-0.04))
    fig_pie.update_annotations(yshift=-10)
    return fig_pie

@app.callback(
    Output('bar-main', 'figure'),
    Input('map-main', 'hoverData'))
def update_bar(hoverdata):    
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


if __name__ == '__main__':
    app.run_server()