"""JSONL event logger for granular run audit trails."""

import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from src.db import log_run_event

LOGS_DIR = Path("logs")


class RunLogger:
    """Session logger appending events to logs/run_<run_id>.jsonl."""

    def __init__(self, run_id: Optional[str] = None):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self.log_file = LOGS_DIR / f"{self.run_id}.jsonl"
        self._step_start_times: Dict[str, float] = {}

    def start_step(self, step_name: str):
        """Mark start time for latency tracking."""
        self._step_start_times[step_name] = time.time()

    def log_event(
        self,
        step: str,
        attempt: int,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        tool: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Append a structured JSON line event to the log file."""
        now = datetime.now(timezone.utc).isoformat()
        start = self._step_start_times.get(step, time.time())
        latency_ms = int((time.time() - start) * 1000)

        event = {
            "timestamp": now,
            "run_id": self.run_id,
            "step": step,
            "attempt": attempt,
            "status": status,
            "latency_ms": latency_ms,
            "tool": tool,
            "model": model,
            "details": details or {},
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[Logger] Failed to write event: {e}")
        log_run_event(self.run_id, step, status, {
            **event["details"], "attempt": attempt, "latency_ms": latency_ms,
            "tool": tool, "model": model, "timestamp": now,
        })

    def read_all_events(self) -> list:
        """Read all logged events for this run."""
        if not self.log_file.exists():
            return []
        events = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return events

