"""Health detector for the PC devtunnel HTTP proxy bridge (default 127.0.0.1:8999).

The GPU server reaches 591/ntfy through a Microsoft devtunnel started on the
personal PC; when the PC is powered off nothing listens on the local tunnel
port and the pipeline must fall back to text-only ingestion. Availability is
probed by issuing a lightweight GET *through* the proxy: only a real HTTP 200
counts (connection refused / timeout / tunnel-error status => PC offline).

SSL verification is disabled by default: devtunnel MITMs TLS with its own
certificate chain, which Python's OpenSSL rejects ("Missing Subject Key
Identifier"). Set PROXY_SSL_VERIFY=1 to re-enable once the tunnel CA is
installed in the trust store.
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

DEFAULT_PROXY_URL = os.environ.get("PROXY_URL", "http://127.0.0.1:8999")
# www.591.com.tw answers 200 through the tunnel; img1/img2 roots return 403
# (hotlink rules), so the site root is the reliable liveness target.
PROBE_URL = os.environ.get("PROXY_PROBE_URL", "https://www.591.com.tw/")
VERIFY_SSL = os.environ.get("PROXY_SSL_VERIFY", "0").lower() in ("1", "true", "yes")
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}


def is_proxy_available(proxy_url: str = DEFAULT_PROXY_URL, timeout: int = 3) -> bool:
    """Return True only when `proxy_url` forwards a request that gets HTTP 200."""
    proxies = {"http": proxy_url, "https": proxy_url}
    try:
        resp = requests.get(
            PROBE_URL, proxies=proxies, headers=_HEADERS, timeout=timeout, verify=VERIFY_SSL
        )
        return resp.status_code == 200
    except requests.RequestException as e:
        logger.info("proxy %s unavailable: %s", proxy_url, e)
        return False
