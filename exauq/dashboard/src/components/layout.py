from __future__ import annotations
from typing import TYPE_CHECKING

from dash import html

if TYPE_CHECKING:
    from exauq.dashboard.exadash import ExaDash


def create_layout(app: ExaDash) -> html.Div:
    return html.Div(
        children=[
            html.H1(app.title),
            html.Hr(),
        ]
    )
