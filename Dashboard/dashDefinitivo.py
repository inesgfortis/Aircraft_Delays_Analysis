# Importamos las librerias mínimas necesarias
import dash
from dash import html
import dash_bootstrap_components as dbc

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
    ],fluid=True)

########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    app.run_server(debug=False)