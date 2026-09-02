"""Proxy health detector: only a real HTTP 200 through the tunnel means LIVE."""

import pytest
import requests

from src.utils import proxy_check


class _Resp:
    def __init__(self, status_code=200):
        self.status_code = status_code


def test_live_on_200(monkeypatch):
    seen = {}

    def fake_get(url, **kw):
        seen.update(url=url, **kw)
        return _Resp(200)

    monkeypatch.setattr(proxy_check.requests, "get", fake_get)
    assert proxy_check.is_proxy_available("http://127.0.0.1:8999", timeout=2) is True
    assert seen["proxies"] == {"http": "http://127.0.0.1:8999", "https": "http://127.0.0.1:8999"}
    assert seen["timeout"] == 2


def test_dead_on_non_200(monkeypatch):
    monkeypatch.setattr(proxy_check.requests, "get", lambda url, **kw: _Resp(403))
    assert proxy_check.is_proxy_available() is False


def test_dead_on_connection_error(monkeypatch):
    def boom(url, **kw):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(proxy_check.requests, "get", boom)
    assert proxy_check.is_proxy_available() is False


def test_dead_on_timeout(monkeypatch):
    def boom(url, **kw):
        raise requests.Timeout("tunnel down")

    monkeypatch.setattr(proxy_check.requests, "get", boom)
    assert proxy_check.is_proxy_available() is False
