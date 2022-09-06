from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from enum import Enum

from exauq.utilities.JobStatus import JobStatus

app = Dash(external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])

list_group = dbc.ListGroup(
    [],
    style={"max-height": "calc(100vh - 130px)", "overflow": "scroll", "margin-bottom": "10px"},
)

app_layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H4("Jobs"),
                    className="text-center"
                ),
                dbc.Col(
                    html.H4("Details"),
                    className="text-center"
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(list_group),
                dbc.Col(html.Div(html.P("..."))),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button([html.I(className="bi bi-funnel")], id="tooltip-target-btn-filter"),
                        dbc.Button([html.I(className="bi bi-sort-down")], id="tooltip-target-btn-sort"),
                        dbc.Button([html.I(className="bi bi-three-dots")], id="tooltip-target-btn-more"),
                        dbc.Tooltip("Filter Jobs", target="tooltip-target-btn-filter", placement="bottom"),
                        dbc.Tooltip("Sort Jobs", target="tooltip-target-btn-sort",  placement="bottom"),
                        dbc.Tooltip("More Options", target="tooltip-target-btn-more", placement="bottom"),
                    ],
                    className="d-flex justify-content-center"
                ),
                dbc.Col(html.Div(html.P("...."))),
            ],
        ),
        html.Footer(
            [
                html.Small("IP Address"),
                html.I(className="bi bi-cpu-fill", style={"color": "#03fc9d"}),
            ],
            className="mt-auto d-flex w-100 justify-content-between",
        ),
        filter_modal(),
        html.Div(id='live-update-text'),
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # in milliseconds
            n_intervals=0
        ),
    ],
    className="d-flex flex-column min-vh-100",
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
