# -*- coding: utf-8 -*-
"""Standalone Step 7 FastAPI Web Server Launcher."""
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flight_delay.common.constants import MONITORING_CFG

if __name__ == "__main__":
    uvicorn.run(
        "flight_delay.api.app:app",
        host=str(MONITORING_CFG["serving"]["host"]),
        port=int(MONITORING_CFG["serving"]["port"]),
        reload=False,
    )
