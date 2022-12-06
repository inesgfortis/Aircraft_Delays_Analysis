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
from plotly.subplots import make_subplots


## App
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LUX], use_pages=True)

app.layout = dbc.Container(
    children = [ 
        html.Br(),
        html.H1("US AIR TRAFFIC"),
        html.Hr(),

        dbc.Nav(
                children = [
                    dbc.NavLink([
                        html.Div(page["name"]),
                    ],
                    href=page["path"],
                    active="exact",
                    )
                    for page in dash.page_registry.values()
                ],
                vertical = True,
                pills = True,
        ),


        # dbc.Tabs(
        #     [
        #         dbc.Tab(label="Airports", tab_id="tab_airports", href = page["path"]),
        #         dbc.Tab(label="Prediction", tab_id="tab_forecast"),
        #         dbc.Tab(label="Fines", tab_id="tab_airlines"),
        #     ],
        #     id="tabs",
        #     active_tab="tab_airports",
        # ),
        html.Div(dash.page_container),
    ]
)


########################################################################################################################
# Callbacks
########################################################################################################################

#callback para seleccionar contenido tabs
# @app.callback(
#     Output("tab-content", "children"),
#     Input("tabs", "active_tab"),
# )

# def get_tab_content(active_tab):
#     """
#     This callback takes the 'active_tab' property as input, as well as the
#     stored graphs, and renders the tab content depending on what the value of
#     'active_tab' is.
#     """
#     if active_tab is not None:
#         if active_tab == "tab_airports":
#             return airports_content
#         elif active_tab == "tab_forecast":
#             return forecast_content
#         elif active_tab == "tab_airlines":
#             return airlines_content
#     return "No tab selected"





########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)