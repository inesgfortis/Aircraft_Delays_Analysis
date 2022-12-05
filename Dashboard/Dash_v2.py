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


app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container(id = "main",
    children = [
        #style={'textAlign': 'left', 'color':'#FFFFFF', 'font-size':20},
        # dbc.Card(
        #     [
                html.Br(),
                html.H1("US AIR TRAFFIC"),
                html.Hr(),
                # dbc.Button(
                #     "Regenerate graphs",
                #     color="primary",
                #     id="button",
                #     className="mb-3",
                # ),
                dbc.Tabs(
                    [
                        dbc.Tab(label="Airports", tab_id="tab_airports"),
                        dbc.Tab(label="Prediction", tab_id="tab_forecast"),
                        dbc.Tab(label="Fines", tab_id="tab_fines"),
                    ],
                    id="tabs",
                    active_tab="tab_forecast",
                ),
                html.Div(id="tab-content", className="p-4"),
                
        #     ]
        # )
       
    ]
)



########################################################################################################################
# TABS CONTENT
########################################################################################################################


airports_content = []


# Forecast tab content 
controls_forecast = dbc.Card(
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
        # html.Div(id="airlines-forecast",
        #     children =[
        #         dbc.Label("Airlines"),
        #         dcc.Dropdown(
        #             id="airline",
        #             options = ['American Airlines','South West'],
        #             value="select an airline",
        #         ),
        #     ]
        # ),
        html.Div(id="fechas-forecast",
            children = [
                dbc.Label("Start date"),
                dcc.DatePickerRange(
                    month_format='MMMM Y',
                    end_date_placeholder_text='DD/MM/YY',
                    # Fechas por defecto
                    start_date=date(2016, 1, 1),
                    end_date=date(2016, 1, 15),
                    id="picker-fechas"
                )
            ]
            
        )
    ],
    body=True,
)

fig_forecast =[
   dcc.Graph(id="graph-forecast")
]

forecast_content = [
    dbc.Row(
                [
                    dbc.Col(controls_forecast,md=4),
                    dbc.Col(fig_forecast, md=8),
                ],
                align="center",
            ),

]


# Fine tab content 
tab_fines = []



########################################################################################################################
# FUNCTIONS
########################################################################################################################

def filter_forecast(airport):
    # Recibe un aeropuerto de origen y devuelve el dataframe y modelo corresondiente a dicho aeropuerto
    
    # print(airport)

    # if airport == 'select an airport':
    #     print("vacio")
    #     airport = str(airports_forecast[0])
    #     print(airport)


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
        #hovermode="x unified", 
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
        elif active_tab == "tab_fines":
            return tab_fines
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


########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
