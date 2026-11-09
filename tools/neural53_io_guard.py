"""Retry transient Windows report-file sharing errors around the frozen driver.

The scientific runtime, manifest, protocol, solver, seeds and models stay frozen.
This adapter changes only save/progress I/O and stops new jobs if reporting fails.
"""

from datetime import datetime, timezone
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


VERSION = "windows_report_io_guard_v1"


def retry_permission_io(operation, *, timeout_seconds=30.0, report=None):
    """Retry a sharing/access collision; propagate other errors immediately."""
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    while True:
        try:
            return operation()
        except PermissionError as error:
            attempts += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            if attempts == 1 and report is not None:
                report(f"[IO-RETRY] {type(error).__name__}: {error}")
            time.sleep(min(0.05 * min(attempts, 5), remaining))


def install_guard(campaign, *, timeout_seconds=30.0, report=None):
    if getattr(campaign, "_report_io_guard_installed", False):
        raise RuntimeError("Report I/O guard already installed")
    original_save = campaign.save
    original_progress = campaign.progress
    if report is None:
        report = lambda message: print(message, file=sys.stderr, flush=True)

    @functools.wraps(original_save)
    def guarded_save(*args, **kwargs):
        return retry_permission_io(
            lambda: original_save(*args, **kwargs),
            timeout_seconds=timeout_seconds,
            report=report,
        )

    @functools.wraps(original_progress)
    def guarded_progress(*args, **kwargs):
        try:
            return retry_permission_io(
                lambda: original_progress(*args, **kwargs),
                timeout_seconds=timeout_seconds,
                report=report,
            )
        except Exception as error:
            # The frozen worker calls its first progress() outside its try block.
            # Ensure a reporting failure drains peers instead of losing a worker.
            campaign.STOP.set()
            with campaign.LOCK:
                campaign.ERRORS.append(
                    {"stage": "progress_io", "error": repr(error)}
                )
            raise

    campaign.save = guarded_save
    campaign.progress = guarded_progress
    campaign._report_io_guard_installed = True


def load_campaign(root):
    frozen = root / "solutions/neural53_v1/runtime/neural53_campaign.py"
    spec = importlib.util.spec_from_file_location("neural53_frozen_guarded", frozen)
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    return campaign


def main():
    root = Path(os.environ["CETSP_PROJECT_ROOT"]).resolve()
    campaign = load_campaign(root)
    install_guard(campaign)
    out = root / "solutions/neural53_v1"
    now = datetime.now(timezone.utc)
    maintenance = out / "maintenance"
    maintenance.mkdir(exist_ok=True)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    record = {
        "version": VERSION,
        "started_utc": now.isoformat(),
        "pid": os.getpid(),
        "adapter_sha256": sha(__file__),
        "frozen_driver_sha256": sha(campaign.__file__),
        "protocol_sha256": sha(campaign.PROTOCOL),
        "changes": [
            "Retry report-file PermissionError for at most 30 seconds per operation",
            "Stop new jobs if progress reporting fails persistently",
        ],
        "scientific_settings_changed": False,
    }
    campaign.save(
        maintenance / f"io_guard_launch_{now:%Y%m%dT%H%M%S}_{os.getpid()}.json",
        record,
    )
    campaign.main()


if __name__ == "__main__":
    main()
