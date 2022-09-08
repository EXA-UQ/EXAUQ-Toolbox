from dash import html
import dash_bootstrap_components as dbc

from exauq.utilities.JobStatus import JobStatus


def generate_icon(job_status: JobStatus) -> html.I:
    if job_status == JobStatus.SUBMITTED:
        icon_class = "bi bi-send-check-fill text-success"
    elif job_status == JobStatus.SUBMIT_FAILED:
        icon_class = "bi bi-send-x-fill text-danger"
    elif job_status == JobStatus.IN_QUEUE:
        icon_class = "bi bi-stack text-info"
    elif job_status == JobStatus.FAILED:
        icon_class = "bi bi-x-square-fill text-danger"
    elif job_status == JobStatus.SUCCESS:
        icon_class = "bi bi-check-square-fill text-success"
    else:
        print("ERROR")
        icon_class = "ERROR"

    return html.I(className=icon_class)


def create_job_lgi(sim_id: str, job_data: dict) -> dbc.ListGroupItem:
    """
    Creates a ListGroupItem for a job
    :param sim_id:
    :param job_data:
    :return:
    """
    if job_data["job_status"] == JobStatus.RUNNING:
        status_indicator = dbc.Spinner(color="success", type="grow", size="sm")
    else:
        status_indicator = generate_icon(job_status=job_data["job_status"])

    submit_time = job_data["submit_time"]
    poll_time = job_data["last_poll_time"]

    if poll_time is None:
        poll_time = "-"

    return dbc.ListGroupItem(
        [
            html.H6(sim_id),
            html.Small("Submitted: " + submit_time, className="text-muted"),
            html.Small("Last Polled: " + poll_time, className="text-muted"),
            status_indicator,
        ],
        className="d-flex w-100 justify-content-between",
    )

