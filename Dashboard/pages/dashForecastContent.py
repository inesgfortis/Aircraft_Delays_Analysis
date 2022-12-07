# Importamos las librerias mínimas necesarias
import pandas as pd
import numpy as np
import pickle
import os

import plotly.graph_objects as go
import dash
from dash import Input, Output, html, callback
from datetime import date,datetime, timedelta

import dash_bootstrap_components as dbc
from dash import dcc
import plotly.express as px


## Data Forecast
data_forecast = pd.read_parquet(str(os.getcwd())+"\\Forecast\\forecast_data.parquet")
airports_forecast = list(data_forecast['ORIGIN_AIRPORT'].unique())
airports_name = list((data_forecast['ORIGIN_AIRPORT']+" | "+data_forecast['ORIGIN_AIRPORT_NAME']).unique())
modelo_forecast_1 = pickle.load(open(str(os.getcwd())+str("\\Forecast\\final_model_1.pickle"),'rb'))
modelo_forecast_2 = pickle.load(open(str(os.getcwd())+str("\\Forecast\\final_model_2.pickle"),'rb'))
modelo_forecast_3 = pickle.load(open(str(os.getcwd())+str("\\Forecast\\final_model_3.pickle"),'rb'))
modelo_forecast_4 = pickle.load(open(str(os.getcwd())+str("\\Forecast\\final_model_4.pickle"),'rb'))
modelo_forecast_5 = pickle.load(open(str(os.getcwd())+str("\\Forecast\\final_model_5.pickle"),'rb'))


## Dash
#app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
dash.register_page(__name__, name = "Forecast")


########################################################################################################################
# TAB CONTENT
########################################################################################################################


layout = [
    dbc.Row(dbc.Col(html.H2('FORECAST DELAYED FLIGHTS', className='text-center text-primary, mb-3'))),
    dbc.Row(
        [
            # Filtro aeropuerto
            dbc.Col(
                [
                    dbc.Card([
                        dbc.Label("Airports"),

                        dcc.Dropdown(
                            id="airport",
                            options =[
                                #{"label": airport, "value": airport} for airport in list(data_forecast['ORIGIN_AIRPORT'].unique())
                                {"label": name, "value": airport} for airport,name in zip(airports_forecast,airports_name)
                            ],
                            # Inicialización por defecto:
                            value=str(airports_forecast[0]),
                        ),
                    ],style = {
                        "padding-top": "2%",
                        "padding-left": "4%",
                        "padding-right": "4%",
                        "padding-bottom": "2%",
                        },
                    ),
                ], id="airports-forecast", 
                style = {
                    "width":"70%",
                    "height": "100%",
                    "vertical-align": "center",
                    "padding-left": "2%",
                    "padding-top": "2%",
                },          
            ),
            # Filtro fechas
            dbc.Col(
                [
                    dbc.Card([
                        dbc.Col(dbc.Label("Forecast dates")),
                        dcc.DatePickerRange(
                            id="picker-fechas",
                            month_format='MMMM Y',
                            end_date_placeholder_text='DD/MM/YY',
                            # Fechas por defecto
                            start_date=date(2016, 1, 1),
                            end_date=date(2016, 1, 15),
                            min_date_allowed = "2016-1-1", 
                            max_date_allowed = "2016-1-31",
                        ),
                    ],style = {
                        "padding-top": "2%",
                        "padding-left": "4%",
                        "padding-right": "4%",
                        "padding-bottom": "2%",
                        },
                    
                    ),
                ], id="fechas-forecast",
                style = {
                    "width":"70%",
                    "height": "100%",
                    "vertical-align": "center",
                    "padding-left": "2%",
                    "padding-right": "2%",
                    "padding-top": "2%",
                },          
            ),
            
        ],
    ),

    dbc.Row(
        [
            dcc.Graph(id="graph-forecast"), 
        ], style = {
                "width":"100%",
                "vertical-align": "center",
                "padding-left": "2%",
                "padding-right": "2%",
                "padding-top": "2%",
        },
    
    ),

]

 

########################################################################################################################
# FUNCTIONS
########################################################################################################################

def filter_forecast(airport):
    # 
    """
    Recibe un aeropuerto de origen y devuelve el dataframe y modelo corresondiente a dicho aeropuerto
    Parameters:
      -  airport: str aeropuerto de origen seleccionado en el dash mediante un desplegable

    Output:
      -  data_df: dataframe con los datos correspondientes a dicho aeropuerto
      -  modelo_forecast: mejor modelo dado el aeropuerto seleccionado

    """
 
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

    """
    Parameters:
      -  data_df: dataframe con los datos correspondientes al aeropuerto seleccionado en el filtro del dash
      -  modelo_forecast: mejor modelo dado el aeropuerto seleccionado
      -  start_date, end_date: rango de fechas inicio y fin para el forecast

    Output:
      -  fig: crea un gráfico con los datos históricos del aeropuerto y los datos de forecast para las fechas seleccionadas

    """

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


########################################################################################################################
# Callbacks
########################################################################################################################


# Callback para seleccionar modelo y predecir devolviendo el gráfico de forecast
@callback(
    Output("graph-forecast", "figure"),
    Input('airport', 'value'),
    Input('picker-fechas', 'start_date'),
    Input('picker-fechas', 'end_date')
)

def update_forecast_graph(airport,start_date,end_date):
    """
    Selecciona los datos y el modelo del aeropuerto mediante la función filter_forecast, se los pasa a la función
    encargada de representar el gráfico con los datos históricos y el forecast y los muestra

    Parameters:
      -  airport: str aeropuerto de origen seleccionado en el dash mediante un desplegable
      -  start_date, end_date: rango de fechas inicio y fin para el forecast

    Output:
      -  go.Figure(): scatter que muestra los datos históricos del aeropuerto y los datos de forecast para las fechas seleccionadas

    """

    data_df,modelo_forecast = filter_forecast(airport)

    if (data_df.size > 0):
        return graph_figure_forecast(data_df,modelo_forecast,start_date,end_date)

    return go.Figure()


########################################################################################################################
########################################################################################################################