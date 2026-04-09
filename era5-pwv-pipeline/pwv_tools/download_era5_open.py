#!/usr/bin/env python

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from .era5_download_core import Era5DownloadConfig, run_era5_download

AUTHOR = "Zhenyi Zhang"
CONTACT = "zhenyzhang@ethz.ch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download hourly ERA5 pressure-level GRIB files."
    )
    parser.add_argument("--start", required=True, help="Start time in YYYY-mm-dd-HH")
    parser.add_argument("--end", required=True, help="End time in YYYY-mm-dd-HH")
    parser.add_argument("--output-root", required=True, help="Root directory for downloaded GRIB files")
    parser.add_argument("--log-dir", default=None, help="Optional log directory")
    parser.add_argument("--task-name", default="ERA5_download", help="Task label used in progress logs")
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Enable monthly email notifications via ERA5_SMTP_* environment variables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Era5DownloadConfig(
        start_time=dt.datetime.strptime(args.start, "%Y-%m-%d-%H"),
        end_time=dt.datetime.strptime(args.end, "%Y-%m-%d-%H"),
        output_root=Path(args.output_root),
        log_dir=Path(args.log_dir) if args.log_dir else None,
        task_name=args.task_name,
        monthly_notification=args.notify,
    )

    notifier = None
    if args.notify:
        from .download_era5_helpers_v2 import send_status_email

        notifier = send_status_email

    run_era5_download(config, notifier=notifier)


if __name__ == "__main__":
    main()
