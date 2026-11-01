"""Persistent conservative GPU accounting, forecasts and a hard job watchdog."""
from __future__ import annotations

import math
import os
from pathlib import Path
import threading
import time
from typing import Callable

from tier1a.common import StepFailure, read_json, write_json


class ResourceBudget:
    """Charge allocated devices for wall time, including setup and CPU work.

    Reserve the entire permitted job before starting. A killed job is charged its
    reservation on restart; normal completion settles to measured wall time.
    """
    def __init__(self, path: Path, *, devices: int, prior_gpu_hours: float = 0,
                 clock: Callable[[], float] = time.monotonic, job_hours: float = 6):
        if (devices < 0 or not math.isfinite(prior_gpu_hours) or not 0 <= prior_gpu_hours <= 60
                or not math.isfinite(job_hours) or not 0 < job_hours <= 6):
            raise StepFailure("budget", "Invalid devices, prior GPU-hours or job limit")
        self.path, self.devices, self.clock = path, devices, clock
        self.started = clock()
        self.cpu_started = time.process_time()
        self.timer: threading.Timer | None = None
        self.data = read_json(path) if path.exists() else {
            "schema_version": 1, "gpu_hours": 0.0, "cpu_wall_hours": 0.0,
            "prior_gpu_hours": prior_gpu_hours, "jobs": [], "ceiling_gpu_hours": 60}
        if prior_gpu_hours < self.data["prior_gpu_hours"]:
            raise StepFailure("budget", "Prior pilot usage cannot decrease on resume")
        self.data["prior_gpu_hours"] = prior_gpu_hours
        active = self.data.pop("active", None)
        if active:
            self.data["gpu_hours"] += active["reserved_gpu_hours"]
            self.data["jobs"].append(dict(active, status="INTERRUPTED_CHARGED_RESERVATION"))
            write_json(path, self.data)
        available = 60 - self.data["prior_gpu_hours"] - self.data["gpu_hours"]
        self.seconds = min(job_hours * 3600, available * 3600 / devices if devices else job_hours * 3600)
        # Leave shutdown margin; fail before consuming the ceiling.
        if self.seconds <= 60:
            raise StepFailure("budget", "No job allowance remains within the 60 GPU-hour pilot ceiling")
        self.data["active"] = {"reserved_gpu_hours": self.seconds * devices / 3600,
                               "devices": devices, "job_limit_seconds": self.seconds}
        write_json(path, self.data)

    def arm(self, failure_report: Path) -> None:
        def stop() -> None:
            # Reservation remains charged even if inference cannot be interrupted cleanly.
            try:
                report = read_json(failure_report) if failure_report.exists() else {}
                report.update(status="FAIL", step1_complete=False, step="budget_watchdog",
                              failing_step="budget_watchdog", reason="Stopped before job/GPU allowance expired")
                write_json(failure_report, report)
            finally:
                os._exit(2)
        self.timer = threading.Timer(max(0, self.seconds - (self.clock() - self.started) - 60), stop)
        self.timer.daemon = True
        self.timer.start()

    def check(self, remaining: int, seconds_per_image: float = 15, setup_seconds: float = 0) -> dict:
        elapsed = self.clock() - self.started
        projected_seconds = remaining * seconds_per_image * 1.5 + setup_seconds
        projected_total = self.data["prior_gpu_hours"] + self.data["gpu_hours"] + (
            elapsed + projected_seconds) * self.devices / 3600
        forecast = {"remaining_images": remaining, "seconds_per_image": seconds_per_image,
                    "safety_factor": 1.5, "projected_pilot_gpu_hours": projected_total,
                    "elapsed_job_seconds": elapsed, "projected_remaining_seconds": projected_seconds}
        self.data["projection"] = forecast
        write_json(self.path, self.data)
        if projected_total > 60 or elapsed + projected_seconds >= self.seconds - 60:
            raise StepFailure("budget_projection", f"Projected usage exceeds allowance: {forecast}")
        return forecast

    def finish(self, status: str) -> None:
        if self.timer:
            self.timer.cancel()
        elapsed = self.clock() - self.started
        active = self.data.pop("active")
        measured = elapsed * self.devices / 3600
        self.data["gpu_hours"] += measured
        self.data["cpu_wall_hours"] += elapsed / 3600 if not self.devices else 0
        self.data["cpu_process_hours"] = self.data.get("cpu_process_hours", 0) + (
            time.process_time() - self.cpu_started) / 3600
        self.data["jobs"].append(dict(active, status=status, elapsed_seconds=elapsed, gpu_hours=measured))
        self.data["total_pilot_gpu_hours"] = self.data["prior_gpu_hours"] + self.data["gpu_hours"]
        write_json(self.path, self.data)
