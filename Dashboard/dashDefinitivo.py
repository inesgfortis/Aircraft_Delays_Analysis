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
                pills = True,
        ),

        html.Div(dash.page_container),
    ]
)


########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)