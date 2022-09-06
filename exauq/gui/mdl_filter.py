from dash import html
import dash_bootstrap_components as dbc


def mdl_filter():
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Filter Jobs"), close_button=True),
            dbc.ModalBody(
                html.Div(
                    [
                        html.Div(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Button(
                                                [
                                                    html.I(className="bi bi-send-check-fill")
                                                ],
                                                outline=True,
                                                color="success",
                                                style={"border-radius": "50%"},
                                            )
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                [
                                                    html.I(className="bi bi-send-x-fill")
                                                ],
                                                outline=True,
                                                color="danger",
                                                style={"border-radius": "50%"},
                                            )
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                [
                                                    html.I(className="bi bi-stack")
                                                ],
                                                outline=True,
                                                color="info",
                                                style={"border-radius": "50%"},
                                            )
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                [
                                                    dbc.Spinner(type="grow", size="sm")
                                                ],
                                                outline=True,
                                                color="success",
                                                style={"border-radius": "50%"},
                                            )
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                [
                                                    html.I(className="bi bi-x-square-fill")
                                                ],
                                                outline=True,
                                                color="danger",
                                                style={"border-radius": "50%"},
                                            )
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                [
                                                    html.I(className="bi bi-check-square-fill")
                                                ],
                                                outline=True,
                                                color="success",
                                                style={"border-radius": "50%"},
                                            )
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.P("Submitted", className="text-center", style={"font-size": "12px"})
                                        ),
                                        dbc.Col(
                                            html.P("Submit Failed", className="text-center", style={"font-size": "12px"})
                                        ),
                                        dbc.Col(
                                            html.P("In Queue", className="text-center", style={"font-size": "12px"})
                                        ),
                                        dbc.Col(
                                            html.P("Running", className="text-center", style={"font-size": "12px"})
                                        ),
                                        dbc.Col(
                                            html.P("Failed", className="text-center", style={"font-size": "12px"})
                                        ),
                                        dbc.Col(
                                            html.P("Completed", className="text-center", style={"font-size": "12px"})
                                        ),
                                    ],
                                )
                            ]
                        )
                    ]
                ),
            ),
            dbc.ModalFooter(
                html.Div(
                    dbc.Button(
                        "Apply",
                        id="filter-modal-apply",
                        n_clicks=0,
                    ),
                ),
            ),
        ],
        id="filter-modal",
        centered=True,
        is_open=False,
    )