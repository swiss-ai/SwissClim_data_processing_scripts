from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cdsapi
from tqdm import tqdm

from .download_era5_helpers_v2 import (
    DownloadStats,
    is_complete_file,
    start_download_thread,
    wait_for_remote_result,
    write_log_line,
)

DEFAULT_PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70", "100", "125",
    "150", "175", "200", "225", "250", "300", "350", "400", "450", "500",
    "550", "600", "650", "700", "750", "775", "800", "825", "850", "875",
    "900", "925", "950", "975", "1000",
]

DEFAULT_VARIABLES = [
    "geopotential",
    "specific_humidity",
    "temperature",
]


@dataclass
class Era5DownloadConfig:
    start_time: dt.datetime
    end_time: dt.datetime
    output_root: Path
    task_name: str = "ERA5"
    dataset: str = "reanalysis-era5-pressure-levels"
    variables: list[str] | None = None
    pressure_levels: list[str] | None = None
    expected_file_size_bytes: int | None = None
    max_download_wait_seconds: int = 600
    download_retry_pause_seconds: int = 30
    log_dir: Path | None = None
    monthly_notification: bool = False

    def __post_init__(self) -> None:
        if self.variables is None:
            self.variables = list(DEFAULT_VARIABLES)
        if self.pressure_levels is None:
            self.pressure_levels = list(DEFAULT_PRESSURE_LEVELS)
        if self.log_dir is None:
            self.log_dir = self.output_root.parent


def hourly_datetimes(start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
    values: list[dt.datetime] = []
    cursor = start
    while cursor <= end:
        values.append(cursor)
        cursor += dt.timedelta(hours=1)
    return values


def build_output_path(root: Path, stamp: dt.datetime) -> Path:
    folder = root / f"{stamp.year:04d}" / f"{stamp.month:02d}" / f"{stamp.day:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"era5_global_37_{stamp:%Y%m%d%H}_gst.grib"


def build_request(stamp: dt.datetime, variables: list[str], pressure_levels: list[str]) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "format": "grib",
        "variable": variables,
        "pressure_level": pressure_levels,
        "year": stamp.strftime("%Y"),
        "month": stamp.strftime("%m"),
        "day": stamp.strftime("%d"),
        "time": stamp.strftime("%H:00"),
        "data_format": "grib",
        "download_format": "unarchived",
    }


def format_status_message(
    label: str,
    stats: DownloadStats,
    started_at: float,
    task_name: str,
    current_name: str,
) -> str:
    return (
        f"{label}\n"
        f"Downloaded: {stats.downloaded}\n"
        f"Already existed: {stats.existing}\n"
        f"Failed: {stats.failed}\n"
        f"Elapsed seconds: {time.time() - started_at:.2f}\n"
        f"Current file: {current_name}\n"
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
        f"Task: {task_name}\n"
    )


def run_era5_download(
    config: Era5DownloadConfig,
    notifier: Callable[[str], bool] | None = None,
) -> DownloadStats:
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    log_path = config.log_dir / f"Log_{config.task_name}.txt"
    err_path = config.log_dir / f"Err_{config.task_name}.txt"
    started_at = time.time()
    stats = DownloadStats()
    schedule = hourly_datetimes(config.start_time, config.end_time)

    intro = (
        f"Program start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
        f"Time range: {config.start_time:%Y%m%d%H} to {config.end_time:%Y%m%d%H}"
    )

    with log_path.open("w", encoding="utf-8") as log_file, err_path.open("w", encoding="utf-8") as err_file:
        write_log_line(log_file, intro)
        write_log_line(err_file, intro)

        client = cdsapi.Client(debug=False, wait_until_complete=False)
        pending_download = None
        notified_month = None

        try:
            with tqdm(total=len(schedule), desc=config.task_name) as progress:
                for stamp in schedule:
                    output_path = build_output_path(config.output_root, stamp)
                    data_name = output_path.name

                    if is_complete_file(output_path, config.expected_file_size_bytes):
                        stats.existing += 1
                        write_log_line(log_file, f"{data_name} already exists")
                        progress.update(1)
                        continue

                    result = client.retrieve(
                        config.dataset,
                        build_request(stamp, config.variables, config.pressure_levels),
                    )
                    request_ok = wait_for_remote_result(result, data_name, log_file, err_file)

                    if pending_download is not None and pending_download.is_alive():
                        pending_download.join()

                    if not request_ok:
                        stats.failed += 1
                        progress.update(1)
                        continue

                    pending_download = start_download_thread(
                        result,
                        output_path,
                        log_file,
                        err_file,
                        stats,
                        max_wait_seconds=config.max_download_wait_seconds,
                        retry_pause_seconds=config.download_retry_pause_seconds,
                    )
                    progress.update(1)

                    if notifier is not None and config.monthly_notification and notified_month != stamp.month:
                        notified_month = stamp.month
                        notifier(
                            format_status_message(
                                "ERA5 download is running",
                                stats,
                                started_at,
                                config.task_name,
                                data_name,
                            )
                        )

            if pending_download is not None and pending_download.is_alive():
                pending_download.join()

            summary = (
                f"Program finished successfully in {time.time() - started_at:.2f}s\n"
                f"Time range: {config.start_time:%Y%m%d%H} to {config.end_time:%Y%m%d%H}\n"
                f"Downloaded: {stats.downloaded}\n"
                f"Already existed: {stats.existing}\n"
                f"Failed: {stats.failed}"
            )
            write_log_line(log_file, summary)
            write_log_line(err_file, summary)
            if notifier is not None:
                notifier(format_status_message("ERA5 download finished successfully", stats, started_at, config.task_name, "done"))
            return stats
        except Exception as exc:
            if pending_download is not None and pending_download.is_alive():
                pending_download.join()

            crash_message = (
                f"Program terminated with an error after {time.time() - started_at:.2f}s\n"
                f"Downloaded: {stats.downloaded}\n"
                f"Already existed: {stats.existing}\n"
                f"Failed: {stats.failed}\n"
                f"Error: {exc}"
            )
            write_log_line(err_file, crash_message)
            if notifier is not None:
                notifier(format_status_message("ERA5 download failed", stats, started_at, config.task_name, str(exc)))
            raise
