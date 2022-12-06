# Importamos las librerias mínimas necesarias
import pandas as pd
import numpy as np
import pickle
import os

import plotly.graph_objects as go
import dash
from dash import Input, Output, dcc, html, State
from datetime import date,datetime, timedelta

import dash_bootstrap_components as dbc
import plotly.express as px
import logging

## Forecast
data_forecast = pd.read_parquet(str(os.getcwd())+"\\Forecast\\forecast_data.parquet")
airports_forecast = list(data_forecast['ORIGIN_AIRPORT'].unique())
modelo_forecast_1 = pickle.load(open(str(os.getcwd())+str("\\Forecast\\final_model_1.pickle"),'rb'))
modelo_forecast_2 = pickle.load(open(str(os.getcwd())+str("\\Forecast\\final_model_2.pickle"),'rb'))
modelo_forecast_3 = pickle.load(open(str(os.getcwd())+str("\\Forecast\\final_model_3.pickle"),'rb'))
modelo_forecast_4 = pickle.load(open(str(os.getcwd())+str("\\Forecast\\final_model_4.pickle"),'rb'))
modelo_forecast_5 = pickle.load(open(str(os.getcwd())+str("\\Forecast\\final_model_5.pickle"),'rb'))


## Airlines
data_fines = pd.read_parquet(str(os.getcwd())+"\\AirlinesAnalysis\\dataFines.parquet")
airlines_list = list(data_fines['AIRLINE'].unique())

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container(id = "main",
    children = [
        html.Br(),
        html.H1("US AIR TRAFFIC"),
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(label="Airports", tab_id="tab_airports"),
                dbc.Tab(label="Prediction", tab_id="tab_forecast"),
                dbc.Tab(label="Fines", tab_id="tab_airlines"),
            ],
            id="tabs",
            active_tab="tab_airlines",
        ),
        html.Div(id="tab-content", className="p-4"),
    ]
)



########################################################################################################################
# TABS CONTENT
########################################################################################################################


airports_content = []

# ----------------------------------------------------------------------------------------------------------------------
forecast_content = [
    dbc.Row(
        [
            dbc.Col(dbc.Card(
            [
                html.Div(id="airports-forecast",
                    children = [
                        dbc.Label("Airports"),
                        dcc.Dropdown(
                            id="airport",
                            options =[
                                {"label": airport, "value": airport} for airport in list(data_forecast['ORIGIN_AIRPORT'].unique())
                            ],
                            # Inicialización por defecto:
                            value=str(airports_forecast[0]),
                        ),
                    ]
                ),
                html.Div(id="fechas-forecast",
                    children = [
                        dbc.Label("Start date"),
                        dcc.DatePickerRange(
                            id="picker-fechas",
                            month_format='MMMM Y',
                            end_date_placeholder_text='DD/MM/YY',
                            # Fechas por defecto
                            start_date=date(2016, 1, 1),
                            end_date=date(2016, 1, 15),
                            min_date_allowed = "2015-1-1",
                            
                        )
                    ]  
                ),
            ], body=True,) ,md=4),

            dbc.Col(dcc.Graph(id="graph-forecast"), md=8),
        ],
                align="center",
    ),

]

# ----------------------------------------------------------------------------------------------------------------------
# Airlines tab content
airlines_content = [
    
    dbc.Row(
        dbc.Card(
            [
                dbc.Row(
                [
                    
                    dbc.Col(html.Div(id="airline-filter",
                        children = [
                            dbc.Label("Airlines"),
                            dcc.Dropdown(
                                id="airline",
                                options =[
                                        {"label": airline, "value": airline} for airline in list(data_fines['AIRLINE'].unique())
                                ],
                                # Inicialización por defecto:
                                value=str(airlines_list[0]),
                            ),
                        ]
                    ),md=6),

                    dbc.Col(html.Div(id="fechas-airlines",
                        children = [
                            dbc.Label("Period"),
                            dcc.DatePickerRange(
                                month_format='MMMM Y',
                                end_date_placeholder_text='DD/MM/YY',
                                # Fechas por defecto
                                start_date=date(2015, 1, 1),
                                end_date=date(2015, 12,31),
                                min_date_allowed = "2015-1-1",
                                max_date_allowed = "2015-12-31",
                                id="picker-fechas-airlines"
                            )
                        ]
                    ),md=4),
                ]),
            ],
            body=True,
        )      
    ),html.Br(),
    
    dbc.Row(
        [
            dbc.Col(
                dcc.Graph(id= "graph-airline-traffic"),md=4),
            dbc.Col(
                dcc.Graph(id= "graph-airline-delay"),md=4),
            dbc.Col(
                dcc.Graph(id= "graph-airline-fines"),md=4),
        ],  align="center",
    ),

    dbc.Row(
        [
            dbc.Col(
                dcc.Graph(id="graph-sunburst"),md=6
            ),
            
            dbc.Col(
                #dcc.Graph(id="graph-causes-airlines"),md=6
            ),
        ],  align="center",
    ),


]



########################################################################################################################
# FUNCTIONS
########################################################################################################################

def filter_forecast(airport):
    # Recibe un aeropuerto de origen y devuelve el dataframe y modelo corresondiente a dicho aeropuerto
 
    data_df = data_forecast[data_forecast['ORIGIN_AIRPORT'] == airport]

    if str(airport) == str(airports_forecast[0]):
        return data_df, modelo_forecast_1
    elif str(airport) == str(airports_forecast[1]):
        return data_df, modelo_forecast_2
    elif str(airport) == str(airports_forecast[2]):
        return data_df, modelo_forecast_3
    elif str(airport) == str(airports_forecast[3]):
        return data_df, modelo_forecast_4
    elif str(airport) == str(airports_forecast[4]):
        return data_df, modelo_forecast_5
    else:
        return pd.DataFrame(),None



def graph_figure_forecast(data,model,start_date,end_date):

    start_date = pd.to_datetime(start_date,format = "%Y-%m-%d")
    end_date = pd.to_datetime(end_date,format = "%Y-%m-%d")

    #predicciones
    pred_days = (end_date - start_date).days+1
    predictions = model.predict(start=len(data),end=(len(data)-1)+pred_days,typ = 'levels').rename('Forecast')

    data_pred = pd.DataFrame()
    data_pred['ds'] = [start_date+timedelta(days=d) for d in range((end_date - start_date).days +1)] 
    data_pred['predictions'] = list(predictions)


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['DATE'], y=data['DELAYED_FLIGHTS'], name = "Year 2015", line_color = px.colors.qualitative.Vivid[5],mode='lines'))
    fig.add_trace(go.Scatter(x=data_pred['ds'], y=data_pred['predictions'], name = "Forecast predictions",line_color = px.colors.qualitative.Vivid[3],mode='lines'))

    fig.update_layout(
        title="Airport predictions",
        xaxis_title="Dates",
        yaxis_title="Number of delays",
        legend_title="Leyend",
        template="plotly_dark",
    )
    return fig




def filter_airline_date(airline,start_date,end_date):

    # Preparamos los datos
    start_date = pd.to_datetime(start_date,format = "%Y-%m-%d")
    end_date = pd.to_datetime(end_date,format = "%Y-%m-%d")

    fechas = [start_date+timedelta(days=d) for d in range((end_date - start_date).days +1)]
    data_airline = data_fines[(data_fines['AIRLINE'] == airline)&(data_fines['DATE'].isin(fechas))]
    
    return data_airline

def create_multas_df(fines_df):

    multas = pd.DataFrame(fines_df.groupby(fines_df['AIRLINE'])["FLIGHT_NUMBER"].count())
    multas = multas.rename(columns={"FLIGHT_NUMBER":"TOTAL_FLIGHTS"})
    multas["SHORT_FLIGHTS"] =  fines_df[fines_df["DISTANCE_TYPE"]== "Short"].groupby(fines_df['AIRLINE'])["FLIGHT_NUMBER"].count()
    multas["MID_FLIGHTS"] =  fines_df[fines_df["DISTANCE_TYPE"]== "Mid"].groupby(fines_df['AIRLINE'])["FLIGHT_NUMBER"].count()
    multas["LONG_FLIGHTS"] =  fines_df[fines_df["DISTANCE_TYPE"]== "Long"].groupby(fines_df['AIRLINE'])["FLIGHT_NUMBER"].count()

    multas['FINE'] = fines_df.groupby(fines_df['AIRLINE'])["FINE"].sum()

    multas["FINE_SHORT"] =  fines_df[fines_df["DISTANCE_TYPE"]== "Short"].groupby(fines_df['AIRLINE'])["FINE"].sum()
    multas["FINE_MID"] =  fines_df[fines_df["DISTANCE_TYPE"]== "Mid"].groupby(fines_df['AIRLINE'])["FINE"].sum()
    multas["FINE_LONG"] =  fines_df[fines_df["DISTANCE_TYPE"]== "Long"].groupby(fines_df['AIRLINE'])["FINE"].sum()

    # Una columna para repartir las multas en función del tiempo retrasado. Type I = (0-30mins) Type II = >1h
    multas["FINE_SHORT_I"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Short') & (fines_df['DELAY_TYPE'] == '(30-60)mins')].groupby(fines_df['AIRLINE'])["FINE"].sum()
    multas["FINE_SHORT_II"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Short') & (fines_df['DELAY_TYPE'] == '>1h')].groupby(fines_df['AIRLINE'])["FINE"].sum()
    multas["FINE_MID_I"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Mid') & (fines_df['DELAY_TYPE'] == '(30-60)mins')].groupby(fines_df['AIRLINE'])["FINE"].sum()
    multas["FINE_MID_II"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Mid') & (fines_df['DELAY_TYPE'] == '>1h')].groupby(fines_df['AIRLINE'])["FINE"].sum()
    multas["FINE_LONG_I"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Long') & (fines_df['DELAY_TYPE'] == '(30-60)mins')].groupby(fines_df['AIRLINE'])["FINE"].sum()
    multas["FINE_LONG_II"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Long') & (fines_df['DELAY_TYPE'] == '>1h')].groupby(fines_df['AIRLINE'])["FINE"].sum()

    # Cuantos vuelos se han retrasado de cada tipo
    multas["SHORT_DELAYED_I"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Short') & (fines_df['DELAY_TYPE'] == '(30-60)mins')].groupby(fines_df['AIRLINE'])["FLIGHT_NUMBER"].count()
    multas["SHORT_DELAYED_II"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Short') & (fines_df['DELAY_TYPE'] == '>1h')].groupby(fines_df['AIRLINE'])["FLIGHT_NUMBER"].count()
    multas["MID_DELAYED_I"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Mid') & (fines_df['DELAY_TYPE'] == '(30-60)mins')].groupby(fines_df['AIRLINE'])["FLIGHT_NUMBER"].count()
    multas["MID_DELAYED_II"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Mid') & (fines_df['DELAY_TYPE'] == '>1h')].groupby(fines_df['AIRLINE'])["FLIGHT_NUMBER"].count()
    multas["LONG_DELAYED_I"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Long') & (fines_df['DELAY_TYPE'] == '(30-60)mins')].groupby(fines_df['AIRLINE'])["FLIGHT_NUMBER"].count()
    multas["LONG_DELAYED_II"] =  fines_df[(fines_df['DISTANCE_TYPE'] == 'Long') & (fines_df['DELAY_TYPE'] == '>1h')].groupby(fines_df['AIRLINE'])["FLIGHT_NUMBER"].count()

    multas = multas.fillna(0)
    multas = multas.reset_index(level=0, drop=False)
  
    return multas


def graph_traffic_airline(data_airline):

    # Obtenemos los datos
    airline_df = pd.DataFrame(data_airline.groupby(data_airline['AIRLINE'])["FLIGHT_NUMBER"].count())
    airline_df = airline_df.rename(columns={"FLIGHT_NUMBER":"TOTAL_FLIGHTS"})
    airline_df["DELAYED_FLIGHTS"] =  data_airline[data_airline["ARRIVAL_DELAY"]>0].groupby(data_airline['AIRLINE'])["FLIGHT_NUMBER"].count()
    airline_df = airline_df.sort_values('TOTAL_FLIGHTS',ascending=False)
    airline_df = airline_df.reset_index(level=0, drop=False)

    # Representamos
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=airline_df['AIRLINE'],
        y=airline_df['TOTAL_FLIGHTS'],
        name='Total Flights',
        marker_color=px.colors.qualitative.Vivid[5]
    ))
    fig.add_trace(go.Bar(
        x=airline_df['AIRLINE'],
        y=airline_df['DELAYED_FLIGHTS'],
        name='Delayed Flights',
        marker_color=px.colors.qualitative.Vivid[9]
    ))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(
        title="Airline Air Traffic Airline",
        xaxis_title="Airlines",
        yaxis_title="Air traffic",
        legend_title="Leyend",
        template="plotly_dark",
        barmode='group', 
        xaxis_tickangle=-45 ,
    )

    fig.update_layout(barmode='group', xaxis_tickangle=-45)
    return fig


def graph_delays_airline(data_airline):
    level_count = pd.DataFrame(data_airline.groupby("AIRLINE")["DELAY_TYPE"].value_counts()).rename(columns = {"DELAY_TYPE": "count"}).reset_index()

    fig = px.histogram(level_count, x="AIRLINE", y="count",barnorm='percent', color="DELAY_TYPE",text_auto='.2f',
                    title="Flights Distribution per Airline", color_discrete_sequence=px.colors.qualitative.Vivid, template="plotly_dark")

    fig.update_layout(
        title="Flights Arrival Distribution",
        xaxis_title="Airlines",
        yaxis_title="% of flights per arrival type",
        legend_title="Leyend",
        template="plotly_dark",
        hovermode="x unified"
        
    )

    return fig

def graph_fines_airline(data_airline):
    level_count = pd.DataFrame(data_airline.groupby("AIRLINE")["DELAY_TYPE"].value_counts()).rename(columns = {"DELAY_TYPE": "count"}).reset_index()
    level_count = level_count.loc[(level_count['DELAY_TYPE'] == '(30-60)mins') | (level_count['DELAY_TYPE'] == ">1h")]

    fig = px.histogram(level_count, x="AIRLINE", y="count", color="DELAY_TYPE",text_auto='.2f',
                    title="Flights Distribution per Airline", color_discrete_sequence=px.colors.qualitative.Vivid[2:], template="plotly_dark")

    fig.update_layout(
        title="Late arrival flights distribution",
        xaxis_title="Airlines",
        yaxis_title="Amount of flights per delay type",
        legend_title="Leyend",
        template="plotly_dark",
        hovermode="x unified"
        
    )

    return fig




def graph_sunburst_airline(data_airline):

    fig =go.Figure(go.Sunburst(
    
        labels=["Fines", "Short Flights", "S.(0-30 mins)","S.>1h",
                "Mid Flights", "M.(0-30 mins)","M.>1h",
                "Long Flights", "L.(0-30 mins)","L.>1h"],
        
        parents=["","Fines","Short Flights","Short Flights",
                    "Fines","Mid Flights","Mid Flights",
                    "Fines","Long Flights", "Long Flights"],

        values = [data_airline['FINE'].sum()]+[data_airline['FINE_SHORT'].sum()]+[data_airline['FINE_SHORT_I'].sum()]+[data_airline['FINE_SHORT_II'].sum()]+
             [data_airline['FINE_MID'].sum()]+[data_airline['FINE_MID_I'].sum()]+[data_airline['FINE_MID_II'].sum()]+
             [data_airline['FINE_LONG'].sum()]+[data_airline['FINE_LONG_I'].sum()]+[data_airline['FINE_MID_II'].sum()],

        marker = dict(colors=["silver","paleturquoise","paleturquoise","paleturquoise",
                              "yellowgreen","yellowgreen","yellowgreen",
                              "mediumseagreen","mediumseagreen","mediumseagreen"]),
    ))

    fig.update_layout(
        title="Fines distribution by distance and delay",
        template="plotly_dark",
        margin = dict(t=60, l=0, r=0, b=0),
        font_size=14,
    )

    return fig



########################################################################################################################
# Callbacks
########################################################################################################################

#callback para seleccionar contenido tabs
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)

def get_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab is not None:
        if active_tab == "tab_airports":
            return airports_content
        elif active_tab == "tab_forecast":
            return forecast_content
        elif active_tab == "tab_airlines":
            return airlines_content
    return "No tab selected"


#callback para seleccionar modelo y predecir devolviendo el gráfico de forecast
@app.callback(
    Output("graph-forecast", "figure"),
    Input('airport', 'value'),
    Input('picker-fechas', 'start_date'),
    Input('picker-fechas', 'end_date')
)

def update_forecast_graph(airport,start_date,end_date):
    """

    """

    data_df,modelo_forecast = filter_forecast(airport)

    if (data_df.size > 0):
        return graph_figure_forecast(data_df,modelo_forecast,start_date,end_date)

    return go.Figure()


@app.callback(
    Output("graph-airline-traffic", "figure"),
    Input('airline', 'value'),
    Input('picker-fechas-airlines', 'start_date'),
    Input('picker-fechas-airlines', 'end_date')
)

def update_airline_traffic_graph(airline,start_date,end_date):

    """

    """
    data_df = filter_airline_date(airline,start_date,end_date)

    if (data_df.size > 0):
        #return graph_traffic_airline(data_df,start_date,end_date)
        return graph_traffic_airline(data_df)

    return go.Figure()

@app.callback(
    Output("graph-airline-delay", "figure"),
    Input('airline', 'value'),
    Input('picker-fechas-airlines', 'start_date'),
    Input('picker-fechas-airlines', 'end_date')
)

def update_delays_airline(airline,start_date,end_date):
    data_df = filter_airline_date(airline,start_date,end_date)

    if (data_df.size > 0):
        return graph_delays_airline(data_df)

    return go.Figure()

#----------------------------------------------------------------------
@app.callback(
    Output("graph-airline-fines", "figure"),
    Input('airline', 'value'),
    Input('picker-fechas-airlines', 'start_date'),
    Input('picker-fechas-airlines', 'end_date')
)

def update_fines_airline(airline,start_date,end_date):
    data_df = filter_airline_date(airline,start_date,end_date)

    if (data_df.size > 0):
        return graph_fines_airline(data_df)

    return go.Figure()
#----------------------------------------------------------------------

@app.callback(
    Output("graph-sunburst", "figure"),
    Input('airline', 'value'),
    Input('picker-fechas-airlines', 'start_date'),
    Input('picker-fechas-airlines', 'end_date')
)

def update_sunburst_graph(airline,start_date,end_date):
    """

    """
    airline_df_filtered = filter_airline_date(airline,start_date,end_date)
    multas_df =  create_multas_df(airline_df_filtered)


    if (multas_df.size > 0):
        return graph_sunburst_airline(multas_df)

    return go.Figure()


########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
