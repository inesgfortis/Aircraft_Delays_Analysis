

# Importamos las librerias mínimas necesarias
import os
import dash
from dash import Input, Output, html, callback, ctx
import dash_bootstrap_components as dbc
from PIL import Image


## Dash
dash.register_page(__name__,name = "Airlines")


########################################################################################################################
# TAB CONTENT
########################################################################################################################

layout = [
    dbc.Row(dbc.Col(html.H2('Fine calculator', className='text-center text-primary, mb-3'))),  # header row
    
    dbc.Row(
        dbc.CardGroup(
        [
            # Short flights card
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Short Flights", className="card-title",style={"color":"green"}),
                        html.P(html.Li("Delay <30mins: $0")),
                        html.P(html.Li("Delay [30-60]min: $5,000")), 
                        html.P(html.Li("Delay >1h: $7,500")),  
                        #dbc.Button("Calculate", color="success", className="mt-auto"),
                    ]
                )
            ),

            # Mid flights card
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Mid Flights", className="card-title", style={"color":"#ffc107"}),
                        html.P(html.Li("Delay <30mins: $0")),
                        html.P(html.Li("Delay [30-60]min: $10,000")), 
                        html.P(html.Li("Delay >1h: $20,000")),  
                        #dbc.Button("Calculate", color="warning", className="mt-auto"),
                    ]
                )
            ),

            # Long flights card 
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Long flights", className="card-title",style={"color":"red"}),
                        html.P(html.Li("Delay <30mins: $0")),
                        html.P(html.Li("Delay [30-60]min: $20,000")), 
                        html.P(html.Li("Delay >1h: $40,000")),  
                        #dbc.Button("Calculate", color="danger", className="mt-auto"),
                    ], 
                ),
            ), 
        ])
    ),

    dbc.Row([
        dbc.Col(
            html.Img(src=Image.open(str(os.getcwd())+str('\Images\Calculator.png'))),
            
                style = {
                    "width":"20%",
                    "vertical-align": "center",
                    "padding-top": "4%",
                },
            md=4),

        dbc.Col([
            dbc.Row([
                dbc.Col(
                    [
                        html.P("Expected [30-60]min delayed flights"),
                        dbc.Input(type="number", value =0, min=0, step=1, id="delayed-flighs-type-I",),
        
                    ],

                    
                ), 

                dbc.Col(
                    [
                        html.P("Expected [<1h] delayed flights"),
                        dbc.Input(type="number", value =0,min=0,step=1,id="delayed-flighs-type-II"),
                    ],
                    
                ),
                
                
            ], style = {
                        "width":"70%",
                        "vertical-align": "center",
                        "padding-top": "4%",
                },
            
            
            ),

            dbc.Row(
                [
                    dbc.Col([dbc.Button("Calculate", color="success", className="mt-auto",id="short-button",n_clicks=0)]),
                    dbc.Col([dbc.Button("Calculate", color="warning", className="mt-auto",id="mid-button",n_clicks=0)]),
                    dbc.Col([dbc.Button("Calculate", color="danger", className="mt-auto", id="long-button",n_clicks=0)]),
                ],
                style = {
                        "width":"70%",
                        "vertical-align": "center",
                        "padding-top": "4%",
                },
            
            ),

            dbc.Row(
                [
                    dbc.Col([html.H5(id="amount-due"),]),
                    dbc.Col([html.H5(id="Reimbursement:-due"),]),
                    #dbc.Col([dbc.Button("Calculate", color="warning", className="mt-auto")]),
                ],
                style = {
                        "width":"70%",
                        "vertical-align": "center",
                        "padding-top": "4%",
                },
            ),
        ]),
    ])
]



########################################################################################################################
# CALLBACKS
########################################################################################################################

# Callback para cel cálculo de la multa
@callback(
    Output('amount-due', 'children'),
    Input('short-button', 'n_clicks'),
    Input('mid-button', 'n_clicks'),
    Input('long-button', 'n_clicks'),
    Input('delayed-flighs-type-I', 'value'),
    Input('delayed-flighs-type-II', 'value')
)
def displayFine(btn1, btn2, btn3,delays_type_I,delays_type_II):
    """
    Parameters:
      -  btn1, btn2, btn3: corresponden con los botones para calcular la multa el función del tipo de trayecto
      -  delays_type_I,delays_type_II: int numero de vuelos retrasados. Type I entre (30-60)mins y Type II >1h

    Output:
      -  msg: str indica el importe de la multa en función de la distancia y tiempo de retraso y el número de retrasos

    """
    
    if "short-button" == ctx.triggered_id:
        fine = 5000*delays_type_I+7500*delays_type_II
    elif "mid-button" == ctx.triggered_id:
        fine = 10000*delays_type_I+20000*delays_type_II

    elif "long-button" == ctx.triggered_id:
        fine = 20000*delays_type_I+40000*delays_type_II
    else:
        fine = 0

    msg = "Amount due: $"+'{:,}'.format(fine)
    return msg
