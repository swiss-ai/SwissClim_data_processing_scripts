from __future__ import annotations

import os
import smtplib
import threading
import time
from dataclasses import dataclass
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path


@dataclass
class DownloadStats:
    downloaded: int = 0
    existing: int = 0
    failed: int = 0


def send_status_email(message: str) -> bool:
    smtp_user = os.getenv("ERA5_SMTP_USER")
    smtp_pass = os.getenv("ERA5_SMTP_PASS")
    smtp_to = os.getenv("ERA5_NOTIFY_TO")
    smtp_host = os.getenv("ERA5_SMTP_HOST", "smtp.qq.com")
    smtp_port = int(os.getenv("ERA5_SMTP_PORT", "465"))

    if not smtp_user or not smtp_pass or not smtp_to:
        return False

    email = MIMEText(message, "plain", "utf-8")
    email["From"] = f"ERA5 downloader <{smtp_user}>"
    email["To"] = smtp_to
    email["Subject"] = Header("ERA5 download update", "utf-8")

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as client:
        client.login(smtp_user, smtp_pass)
        client.sendmail(smtp_user, [smtp_to], email.as_string())
    return True


def write_log_line(handle, message: str) -> None:
    handle.write(message + "\n")
    handle.flush()


def is_complete_file(path: Path, expected_size_bytes: int | None) -> bool:
    if not path.exists():
        return False
    if expected_size_bytes is None:
        return path.stat().st_size > 0
    return path.stat().st_size == expected_size_bytes


def wait_for_remote_result(result, data_name: str, log_file, err_file, max_update_retries: int = 3) -> bool:
    update_failures = 0
    wait_seconds = 3
    start_time = time.time()

    while True:
        try:
            result.update()
            reply = result.reply
            state = reply["state"]
        except Exception as exc:
            update_failures += 1
            write_log_line(
                err_file,
                f"{data_name} update failed ({update_failures}/{max_update_retries}) after {time.time() - start_time:.2f}s: {exc}",
            )
            if update_failures >= max_update_retries:
                return False
            time.sleep(5)
            continue

        if state == "completed":
            write_log_line(log_file, f"{data_name} request completed after {time.time() - start_time:.2f}s")
            return True

        if state in {"accepted", "queued", "running"}:
            time.sleep(wait_seconds)
            wait_seconds = min(int(wait_seconds * 1.5), 60)
            continue

        error_message = reply.get("error", {}).get("message", "unknown error")
        error_reason = reply.get("error", {}).get("reason", "unknown reason")
        write_log_line(
            err_file,
            f"{data_name} request failed after {time.time() - start_time:.2f}s: {state} | {error_message} | {error_reason}",
        )
        return False


def download_result_file(
    result,
    target_path: Path,
    log_file,
    err_file,
    max_wait_seconds: int = 600,
    retry_pause_seconds: int = 30,
) -> bool:
    start_time = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            result.download(str(target_path))
            write_log_line(log_file, f"{target_path.name} downloaded in {time.time() - start_time:.2f}s")
            return True
        except Exception as exc:
            elapsed = time.time() - start_time
            if elapsed >= max_wait_seconds:
                write_log_line(
                    err_file,
                    f"{target_path.name} download failed after {attempt} attempts and {elapsed:.2f}s: {exc}",
                )
                return False
            write_log_line(
                err_file,
                f"{target_path.name} download retry {attempt} after {elapsed:.2f}s: {exc}",
            )
            time.sleep(retry_pause_seconds)


def start_download_thread(
    result,
    target_path: Path,
    log_file,
    err_file,
    stats: DownloadStats,
    max_wait_seconds: int,
    retry_pause_seconds: int,
) -> threading.Thread:
    def _worker() -> None:
        if download_result_file(
            result,
            target_path,
            log_file,
            err_file,
            max_wait_seconds=max_wait_seconds,
            retry_pause_seconds=retry_pause_seconds,
        ):
            stats.downloaded += 1
        else:
            stats.failed += 1

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread
