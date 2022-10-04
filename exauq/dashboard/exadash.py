from __future__ import annotations
from typing import TYPE_CHECKING

from dash import Dash

from .src.components.layout import create_layout

if TYPE_CHECKING:
    from multiprocessing import connection


class ExaDash(Dash):
    def __init__(self):
        super().__init__()
        self.schedular_connection = None


def run_dashboard(schedular_connection: connection.Connection):
    app = ExaDash()
    app.schedular_connection = schedular_connection
    app.layout = create_layout(app=app)
    app.run(debug=True, use_reloader=False)
