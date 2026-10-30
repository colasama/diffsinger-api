"""Source adapters against a local fake server shaped like each upstream API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from dsapi import sources

LRC = "[00:01.00]别把夏天关掉\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, ctype="application/json", status=200, headers=()):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        base = f"http://{self.headers['Host']}"
        song = {"id": 42, "name": "别把夏天关掉", "ar": [{"name": "乐小君"}, {"name": "用户音源戊"}], "dt": 200591}
        if u.path == "/media.mp3":
            return self._send(b"ID3fake", "audio/mpeg")
        # ncm-api
        if u.path == "/cloudsearch":
            return self._send({"code": 200, "result": {"songs": [song]}})
        if u.path == "/song/detail":
            return self._send({"code": 200, "songs": [song]})
        if u.path == "/lyric":
            return self._send({"code": 200, "lrc": {"lyric": LRC}})
        if u.path == "/song/url/v1":
            assert q["id"] == "42"
            return self._send({"code": 200, "data": [{"id": 42, "url": base + "/media.mp3", "freeTrialInfo": None}]})
        # meting
        if u.path == "/meting/":
            item = {"name": "别把夏天关掉", "artist": "乐小君 / 用户音源戊",
                    "url": f"{base}/meting/?server={q['server']}&type=url&id=42",
                    "lrc": f"{base}/meting/?server={q['server']}&type=lrc&id=42"}
            if q["type"] in ("search", "song"):
                return self._send([item])
            if q["type"] == "lrc":
                return self._send(LRC, "text/plain")
            if q["type"] == "url":
                return self._send(b"", "text/plain", 302, [("Location", base + "/media.mp3")])
        self._send({"code": 404}, status=404)


@pytest.fixture(scope="module")
def server():
    # httpx also honours the macOS system proxy, which would swallow loopback calls.
    mp = pytest.MonkeyPatch()
    mp.setenv("NO_PROXY", "127.0.0.1,localhost")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    mp.undo()


@pytest.mark.parametrize("make", [lambda b: sources.NcmApi(b), lambda b: sources.Meting(b + "/meting", "tencent")])
def test_source_roundtrip(server, make, tmp_path):
    src = make(server)
    hits = src.search("夏天")
    assert hits[0].id == "42" and hits[0].name == "别把夏天关掉"
    assert hits[0].artists == ["乐小君", "用户音源戊"]
    assert src.song_info("42").name == "别把夏天关掉"
    assert src.lyric("42") == LRC
    assert src.download_audio("42", tmp_path / "a").read_bytes() == b"ID3fake"


def test_get_source_requires_url(monkeypatch):
    monkeypatch.setenv("DSAPI_SOURCE", "meting")
    monkeypatch.delenv("DSAPI_SOURCE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DSAPI_SOURCE_URL"):
        sources.get_source()
