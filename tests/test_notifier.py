"""ntfy delivery paths: tunnel-first with direct fallback (GPU egress is blocked)."""

import pytest
import requests

from src import notifier


class _Resp:
    def raise_for_status(self):
        pass


@pytest.fixture
def posts(monkeypatch):
    calls = []

    def fake_post(url, data=None, headers=None, timeout=None, proxies=None):
        calls.append(proxies)
        if getattr(fake_post, "fail", set()) and (proxies or "direct") in fake_post.fail:
            raise requests.ConnectionError("blocked")
        return _Resp()

    monkeypatch.setattr(notifier.requests, "post", fake_post)
    return fake_post, calls


def test_alert_prefers_tunnel(posts):
    _fake_post, calls = posts
    assert notifier.send_ntfy_alert(
        {"listing_id": "1", "title": "t", "url": "u"}, 4.0, 3.5, proxy="http://127.0.0.1:8999"
    ) is True
    assert calls == [{"http": "http://127.0.0.1:8999", "https": "http://127.0.0.1:8999"}]


def test_tunnel_failure_falls_back_to_direct(posts):
    fake_post, calls = posts
    fake_post.fail = {"http://127.0.0.1:8999"}
    assert notifier.send_ntfy_alert(
        {"listing_id": "1", "title": "t", "url": "u"}, 4.0, 3.5, proxy="http://127.0.0.1:8999"
    ) is True
    assert calls[-1] is None  # second attempt went direct


def test_proxy_alert_goes_tunnel_first_then_direct(posts):
    fake_post, calls = posts
    assert notifier.send_proxy_request_alert(3, proxy="http://127.0.0.1:8999") is True
    assert calls[0]["http"] == "http://127.0.0.1:8999"

    fake_post.fail = {"http://127.0.0.1:8999", "direct"}  # PC off: neither path works
    calls.clear()
    assert notifier.send_proxy_request_alert(3, proxy="http://127.0.0.1:8999") is False
    assert len(calls) == 2  # tried tunnel, then direct, gave up (queue persists)


def test_below_threshold_sends_nothing(posts):
    _, calls = posts
    assert notifier.send_ntfy_alert({"listing_id": "1"}, 2.0, 3.5, proxy="http://p") is False
    assert calls == []
