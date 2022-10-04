from __future__ import annotations
from typing import TYPE_CHECKING

from dash import html
import dash_bootstrap_components as dbc

from exauq.dashboard.src import ids

if TYPE_CHECKING:
    from exauq.dashboard.exadash import ExaDash


def create_layout(app: ExaDash) -> html.Div:
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
                            dbc.Col(html.Div(id=ids.DIV_JOB_DETAILS, hidden=True)),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                #Job filter, sort, etc options
                            )
                        ]
                    )
                ],
                className="d-flex flex-column min-vh-100",
            )
        ]
    )
