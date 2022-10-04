from __future__ import annotations
from typing import TYPE_CHECKING

from dash import html
import dash_bootstrap_components as dbc

from exauq.dashboard.src import ids

if TYPE_CHECKING:
    from exauq.dashboard.exadash import ExaDash


def render(app: ExaDash) -> html.Div:
    return html.Div(
        [
            dbc.Button([html.I(className="bi bi-funnel")], id=ids.BTN_FILTER_JOBS),
            dbc.Button([html.I(className="bi bi-sort-down")], id=ids.BTN_SORT_JOBS),
            dbc.Button([html.I(className="bi bi-three-dots")], id=ids.BTN_MORE_JOBS),
            dbc.Tooltip("Filter Jobs", target=ids.BTN_FILTER_JOBS, placement="bottom"),
            dbc.Tooltip("Sort Jobs", target=ids.BTN_SORT_JOBS, placement="bottom"),
            dbc.Tooltip("More Options", target=ids.BTN_MORE_JOBS, placement="bottom"),
        ],
        className="d-flex justify-content-center"
    )
