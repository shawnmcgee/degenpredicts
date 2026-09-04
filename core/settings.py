"""Settings shared by every sport package."""
from __future__ import annotations

import os

HTTP_TIMEOUT = float(os.environ.get("DEGEN_HTTP_TIMEOUT", "30"))
HTTP_RETRIES = int(os.environ.get("DEGEN_HTTP_RETRIES", "4"))
