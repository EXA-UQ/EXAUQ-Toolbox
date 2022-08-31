from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from enum import Enum

from exauq.utilities.JobStatus import JobStatus

app = Dash(external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])

list_group = dbc.ListGroup(
    [
        dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.H5("<Job Name>", className="mb-1"),
                        dbc.Spinner(color="success", type="grow", size="sm")
                    ],
                    className="d-flex w-100 justify-content-between",
                ),
                html.P("And some text underneath", className="mb-1"),
                html.Small("Plus some small print.", className="text-muted"),
                html.I(className="bi bi-cpu-fill")
            ],
            href="https://google.com"
        ),
        dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.H5(
                            "This item also has a heading", className="mb-1"
                        ),
                        html.Small("Ok!", className="text-warning"),
                    ],
                    className="d-flex w-100 justify-content-between",
                ),
                html.P("And some more text underneath too", className="mb-1"),
                html.Small(
                    "Plus even more small print.", className="text-muted"
                ),
            ]
        ),

        dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.H6("MET-09876K"),
                        html.I(className="bi bi-cpu-fill", style={"color": "#03fc9d"}, id="sim_type"),
                        dbc.Tooltip("Local Machine", target="sim_type")
                    ],
                    className="d-flex w-100 justify-content-between",
                ),
            ]
        )
    ],
)


app_layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.Div("A single, half-width column"), width=6),
                dbc.Col(list_group)
            ]
        ),
        html.Div(id='live-update-text'),
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # in milliseconds
            n_intervals=0
        ),
    ]
)

app.layout = app_layout

schedular_connection = None


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    if schedular_connection.poll():
        status = schedular_connection.recv()
        return html.Code(str(status))

    return html.H1("Hello")


def start_dash(connection):
    global schedular_connection
    schedular_connection = connection
    print("Staring Dash")
    app.run(debug=True, use_reloader=False)
