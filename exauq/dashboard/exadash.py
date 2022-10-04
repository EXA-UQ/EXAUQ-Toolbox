from __future__ import annotations
from typing import TYPE_CHECKING

from dash import Dash
import dash_bootstrap_components as dbc

from .src.components.layout import create_layout

if TYPE_CHECKING:
    from multiprocessing import connection


class ExaDash(Dash):
    def __init__(self, refresh_interval=5000):
        super().__init__(external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP])
        self.schedular_connection = None
        self.refresh_interval = refresh_interval


def run_dashboard(schedular_connection: connection.Connection):
    app = ExaDash()
    app.schedular_connection = schedular_connection
    app.title = "EXAUQ-Toolbox"
    app._favicon = ""
    app.layout = create_layout(app=app)
    app.run(debug=True, use_reloader=False)
