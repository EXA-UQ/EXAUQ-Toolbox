import argparse
import time


def sleep_for_period(period: float) -> float:
    """
    Simple function which take in a sleep period and run python sleep function
    for that period.

    Parameters
    ----------
    period: float
        Time period to go to sleep for in seconds

    Return
    ------
    Float:
        Measured sleep period in seconds
    """
    start_time = time.time()
    time.sleep(period)
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":

    # Parse command line argument list. See the *help* key in argument lists
    # for details.
    parser = argparse.ArgumentParser(description="Sleep for a specified period.")
    parser.add_argument(
        "-p", "--period", help="period to sleep for (in seconds).", type=float
    )

    # Set variables based on parsed command line argument list
    args = parser.parse_args()
    period = args.period

    timed_sleep_period = sleep_for_period(period=period)

    # Write output to data file
    output_file = open("results.out", "w")
    output_file.write("period={0}\n".format(timed_sleep_period))
    output_file.close()
