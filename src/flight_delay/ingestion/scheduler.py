# -*- coding: utf-8 -*-
"""[STEP 1] Periodic Collection Scheduler Helper."""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from flight_delay.ingestion.collector import collect_all

log = logging.getLogger(__name__)


def run_collection_loop(every_minutes: int = 120, only: Optional[str] = None, max_iterations: Optional[int] = None) -> None:
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        collect_all(only=only)
        if max_iterations is not None and iteration >= max_iterations:
            break
        nxt = datetime.now() + timedelta(minutes=every_minutes)
        log.info("Next FIDS collection scheduled at %s", nxt.strftime("%H:%M:%S"))
        time.sleep(every_minutes * 60)
