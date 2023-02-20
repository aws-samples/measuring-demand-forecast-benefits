# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for checking and logging progress of long-running operations"""
# Python Built-Ins:
from datetime import datetime
import time
from typing import Callable, Optional, TypeVar, Union

# External Dependencies:
from dateutil.relativedelta import relativedelta  # For nice display purposes


TStatus = TypeVar("TStatus")


def polling_spinner(
    fn_poll_result: Callable[[], TStatus],
    fn_is_finished: Callable[[TStatus], bool],
    fn_stringify_result: Optional[Callable[[TStatus], str]] = None,
    fn_eta: Optional[Callable[[TStatus], Optional[Union[str, relativedelta]]]] = None,
    spinner_secs: float = 0.5,
    poll_secs: float = 30,
    timeout_secs: Optional[float] = None,
) -> TStatus:
    """Polling wait with loading spinner and elapsed time indicator (plus optional ETA).

    Displays a spinner, the current stringified status, and the elapsed time since the wait was
    started or the stringified status last changed (as a dateutil.relativedelta down to 1sec
    precision). For example:

    `/ Status: InProgress - AnalyzingData [Since: relativedelta(minutes=+10, seconds=+33)]`

    A new line is generated with reset "Since", every time the status string changes. No limit on
    line length, but generated status string must not contain newlines.

    Parameters
    ----------
    fn_poll_result :
        Zero-argument callable that returns some kind of job status descriptor, may raise error on
        failure.
    fn_is_finished :
        Checks the status object returned by `fn_poll_result` and returns True if job completed,
        False if ongoing, or should raise an error if the job has failed but `fn_poll_result` still
        fetches the status successfully.
    fn_stringify_result :
        Optional status object stringifier for the console output [defaults to str(status)]
    fn_eta :
        Optional function to extract an estimated time remaining from the result object, returning
        either a `dateutil.relativedelta` object or a plain string
    spinner_secs :
        Time to sleep between check cycles. Choosing a divisor of 1s (or rather, `poll_secs`)
        produces nicer-looking updates.
    poll_secs :
        Minimum elapsed time since last poll after which next check cycle will call fn_poll_result
    timeout_secs : Optional
        Optional number of seconds after which to exit the wait raising TimeoutError [default inf]

    Returns
    -------
    status :
        The final result of `fn_poll_result()`
    """
    SPINNER_STATES = ("/", "-", "\\", "|")
    status = fn_poll_result()
    overall_t0 = datetime.now().replace(microsecond=0)
    status_t0 = overall_t0
    poll_t0 = overall_t0
    status_str = fn_stringify_result(status) if fn_stringify_result else str(status)
    if fn_eta:
        eta_raw = fn_eta(status)
        if eta_raw is None:
            eta_str = ""
        else:
            eta_str = f" [ETA: {eta_raw}]"
    else:
        eta_str = ""
    i = 0
    maxlen = 0
    print(f"Initial status: {status_str}")
    while not fn_is_finished(status):
        t = datetime.now()
        if timeout_secs is not None and (t - overall_t0).total_seconds() >= timeout_secs:
            raise TimeoutError(
                "Maximum wait time exceeded: timeout_secs={}, {}".format(
                    timeout_secs, relativedelta(seconds=timeout_secs)
                )
            )
        elif (t - poll_t0).total_seconds() >= poll_secs:
            newstatus = fn_poll_result()
            poll_t0 = t
            newstatus_str = (
                fn_stringify_result(newstatus) if fn_stringify_result else str(newstatus)
            )
            if status_str == newstatus_str:
                print("\r", end="")
            else:
                print("\n", end="")
                status_t0 = t
            status = newstatus
            status_str = newstatus_str
            if fn_eta:
                eta_raw = fn_eta(status)
                if eta_raw is None:
                    eta_str = ""
                else:
                    eta_str = f" [ETA: {eta_raw}]"
        else:
            print("\r", end="")
        i = (i + 1) % len(SPINNER_STATES)
        msgdelta = relativedelta(t, status_t0)
        msgdelta.microseconds = 0  # No need to print out such high resolution
        msg = f"{SPINNER_STATES[i]} Status: {status_str}{eta_str} [Since: {msgdelta}]"
        maxlen = max(maxlen, len(msg))
        msg = msg.ljust(maxlen)
        print(msg, end="")
        time.sleep(spinner_secs)
    print("")
    return status
