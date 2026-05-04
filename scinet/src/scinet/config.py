from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SciNetConfig:
    base_url: str = "http://scinet.openkg.cn"
    api_key: str = ""
    timeout: int = 900


def load_config(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int | None = None,
) -> SciNetConfig:
    return SciNetConfig(
        base_url=(base_url or os.getenv("SCINET_API_BASE_URL") or os.getenv("KG2API_BASE_URL") or "http://scinet.openkg.cn").rstrip("/"),
        api_key=api_key or os.getenv("SCINET_API_KEY") or os.getenv("KG2API_API_KEY") or "",
        timeout=int(timeout or os.getenv("SCINET_TIMEOUT") or 900),
    )
