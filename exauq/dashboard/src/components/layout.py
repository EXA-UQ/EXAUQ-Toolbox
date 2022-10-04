from __future__ import annotations
from typing import TYPE_CHECKING

from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from exauq.dashboard.src import ids
from exauq.dashboard.src.components import job_display_options, lgi_job

if TYPE_CHECKING:
    from exauq.dashboard.exadash import ExaDash


def create_layout(app: ExaDash) -> html.Div:
    @app.callback(Output(ids.LG_JOBS_CONTAINER, "children"),
                  Input(ids.INTERVAL_COMPONENT, "n_intervals"),
                  State(ids.LG_JOBS_CONTAINER, "children"))
    def update_jobs_list(n, jobs_container):
        if app.schedular_connection.poll():
            status = app.schedular_connection.recv()

            list_group_items = []
            for key, value in status.items():
                list_group_items.append(lgi_job.create_job_lgi(sim_id=key, job_data=value))

            return dbc.ListGroup(
                list_group_items,
                id=ids.LG_JOBS,
                style={"max-height": "calc(100vh - 130px)", "overflow": "scroll", "margin-bottom": "10px"},
            )

        return jobs_container

    return html.Div(
        children=[
            html.H1(app.title, className="text-center"),
            html.Hr(),
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H6("Jobs"),
                                className="text-center",
                            ),
                            dbc.Col(
                                html.H6("Details"),
                                className="text-center",
                            ),
                        ],
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.Div(id=ids.LG_JOBS_CONTAINER)),
                            dbc.Col(html.Div(id=ids.DIV_JOB_DETAILS, hidden=False)),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                job_display_options.render(app=app)
                            )
                        ]
                    )
                ],
                className="d-flex flex-column min-vh-100",
            ),
            dcc.Interval(
                id=ids.INTERVAL_COMPONENT,
                interval=app.refresh_interval,
                n_intervals=0,
            ),
        ]
    )
