"""Music sources: search, metadata, LRC and audio download.

Pick one with DSAPI_SOURCE:
  musicbox (default) 枫叶音乐盒 API (optional DSAPI_SOURCE_URL,
           DSAPI_MUSICBOX_SERVER = netease | kuwo | qq)
  netease  direct calls to music.163.com's public endpoints
  ncm-api  a self-hosted NeteaseCloudMusicApi / api-enhanced instance
           (DSAPI_SOURCE_URL, optional DSAPI_NCM_COOKIE for VIP tracks)
  meting   a self-hosted Meting API (DSAPI_SOURCE_URL, DSAPI_METING_SERVER =
           netease | tencent | kugou | kuwo | ...)
Song ids are strings so non-NetEase servers (e.g. QQ mids) fit.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx

UA = {"User-Agent": "Mozilla/5.0"}


@dataclass
class Song:
    id: str
    name: str
    artists: list[str]
    duration_ms: int | None = None


class Source:
    HEADERS = UA

    def search(self, query: str, limit: int = 10) -> list[Song]:
        raise NotImplementedError

    def song_info(self, song_id: str) -> Song:
        raise NotImplementedError

    def lyric(self, song_id: str) -> str:
        raise NotImplementedError

    def audio_url(self, song_id: str) -> str:
        raise NotImplementedError

    def download_audio(self, song_id: str, dest: Path) -> Path:
        with httpx.Client(headers=self.HEADERS, timeout=60, follow_redirects=True) as c, c.stream("GET", self.audio_url(song_id)) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return dest


def _require_lrc(text: str | None) -> str:
    if not text or not text.strip():
        raise RuntimeError("song has no LRC lyrics")
    return text


class NeteaseDirect(Source):
    BASE = "https://music.163.com"
    HEADERS = {"Referer": "https://music.163.com/", **UA}

    def _get(self, path: str, params: dict) -> dict:
        with httpx.Client(headers=self.HEADERS, timeout=20) as c:
            resp = c.get(self.BASE + path, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 200:
            raise RuntimeError(f"netease {path} code={data.get('code')}")
        return data

    def search(self, query, limit=10):
        # The legacy GET /api/search/get ignores non-ASCII queries now; cloudsearch works.
        with httpx.Client(headers=self.HEADERS, timeout=20) as c:
            resp = c.post(self.BASE + "/api/cloudsearch/pc", data={"s": query, "type": 1, "limit": limit, "offset": 0})
        resp.raise_for_status()
        songs = ((resp.json().get("result") or {}).get("songs")) or []
        return [Song(str(s["id"]), s["name"], [a["name"] for a in s["ar"]], s["dt"]) for s in songs]

    def song_info(self, song_id):
        s = self._get("/api/song/detail", {"ids": f"[{song_id}]"})["songs"][0]
        return Song(str(s["id"]), s["name"], [a["name"] for a in s["artists"]], s["duration"])

    def lyric(self, song_id):
        data = self._get("/api/song/lyric/v1", {"id": song_id, "lv": -1, "kv": -1, "tv": -1})
        return _require_lrc((data.get("lrc") or {}).get("lyric"))

    def audio_url(self, song_id):
        item = self._get("/api/song/enhance/player/url", {"ids": f"[{song_id}]", "br": 128000})["data"][0]
        if not item.get("url") or item.get("freeTrialInfo"):
            # Usually VIP-only or region/copyright locked for anonymous callers.
            raise RuntimeError(
                f"无法获取这首歌的完整音频（code={item.get('code')}, fee={item.get('fee')}，可能需要 VIP 或有版权限制）。"
                "可以改用上传音频 + LRC，或配置 DSAPI_SOURCE=ncm-api 并提供登录 cookie。")
        return item["url"]


class NcmApi(Source):
    """NeteaseCloudMusicApi-compatible server, e.g. NeteaseCloudMusicApiEnhanced/api-enhanced."""

    def __init__(self, base: str, cookie: str = ""):
        self.base, self.cookie = base.rstrip("/"), cookie

    def _get(self, path: str, **params) -> dict:
        if self.cookie:
            params["cookie"] = self.cookie
        with httpx.Client(headers=UA, timeout=20) as c:
            resp = c.get(self.base + path, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") not in (200, None):
            raise RuntimeError(f"ncm-api {path} code={data.get('code')}: {data.get('message') or data.get('msg')}")
        return data

    def search(self, query, limit=10):
        songs = (self._get("/cloudsearch", keywords=query, limit=limit).get("result") or {}).get("songs") or []
        return [Song(str(s["id"]), s["name"], [a["name"] for a in s["ar"]], s["dt"]) for s in songs]

    def song_info(self, song_id):
        s = self._get("/song/detail", ids=song_id)["songs"][0]
        return Song(str(s["id"]), s["name"], [a["name"] for a in s["ar"]], s["dt"])

    def lyric(self, song_id):
        return _require_lrc((self._get("/lyric", id=song_id).get("lrc") or {}).get("lyric"))

    def audio_url(self, song_id):
        item = self._get("/song/url/v1", id=song_id, level="standard")["data"][0]
        if not item.get("url"):
            raise RuntimeError(f"no playable url (code={item.get('code')})")
        if item.get("freeTrialInfo"):
            raise RuntimeError("only a trial clip is available (set DSAPI_NCM_COOKIE for VIP tracks)")
        return item["url"]


class Meting(Source):
    """Meting API (?server=&type=search|song|url|lrc&id=). Its song/search
    items carry `url`/`lrc` links back into the API; the id is in their query."""

    def __init__(self, base: str, server: str = "netease"):
        self.base, self.server = base.rstrip("/"), server

    def _url(self, kind: str, ident: str) -> str:
        return str(httpx.URL(self.base + "/", params={"server": self.server, "type": kind, "id": ident}))

    def _json(self, kind: str, ident: str) -> list[dict]:
        with httpx.Client(headers=self.HEADERS, timeout=20, follow_redirects=True) as c:
            resp = c.get(self._url(kind, ident))
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _song(item: dict) -> Song:
        ident = str(item.get("id") or parse_qs(urlparse(item.get("url", "")).query).get("id", [""])[0])
        artist = item.get("artist") or item.get("author") or ""
        artists = artist if isinstance(artist, list) else [a.strip() for a in str(artist).split("/") if a.strip()]
        return Song(ident, item.get("name") or item.get("title") or "", artists)

    def search(self, query, limit=10):
        return [self._song(i) for i in self._json("search", query)[:limit]]

    def song_info(self, song_id):
        items = self._json("song", song_id)
        if not items:
            raise RuntimeError("song not found")
        song = self._song(items[0])
        song.id = song.id or song_id
        return song

    def lyric(self, song_id):
        with httpx.Client(headers=self.HEADERS, timeout=20, follow_redirects=True) as c:
            resp = c.get(self._url("lrc", song_id))
        resp.raise_for_status()
        return _require_lrc(resp.text)

    def audio_url(self, song_id):
        return self._url("url", song_id)  # redirects to the media file


class MusicBox(Meting):
    """枫叶音乐盒: dedicated search routes plus Meting metadata/LRC/audio."""

    DEFAULT_BASE = "https://fy-musicbox-api.mu-jie.cc"
    HEADERS = {**UA, "Referer": "https://mu-jie.cc/", "Origin": "https://mu-jie.cc"}

    def __init__(self, base: str = DEFAULT_BASE, server: str = "netease"):
        if server not in {"netease", "kuwo", "qq"}:
            raise ValueError("DSAPI_MUSICBOX_SERVER must be netease, kuwo or qq")
        self.api_base = base.rstrip("/")
        super().__init__(self.api_base + "/meting", server)

    @staticmethod
    def _items(data) -> list[dict]:
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise RuntimeError("枫叶音乐盒接口返回异常，预期为歌曲列表")
        return data

    def _json(self, kind: str, ident: str) -> list[dict]:
        return self._items(super()._json(kind, ident))

    def search(self, query, limit=10):
        if limit <= 0:
            return []
        with httpx.Client(headers=self.HEADERS, timeout=20, follow_redirects=True) as c:
            resp = c.get(self.api_base + f"/{self.server}/search/song/",
                         params={"keywords": query, "pn": 1, "limit": limit})
        resp.raise_for_status()
        return [self._song(item) for item in self._items(resp.json())[:limit]]


def get_source() -> Source:
    kind = os.environ.get("DSAPI_SOURCE", "musicbox")
    url = os.environ.get("DSAPI_SOURCE_URL", "")
    if kind == "musicbox":
        return MusicBox(url or MusicBox.DEFAULT_BASE, os.environ.get("DSAPI_MUSICBOX_SERVER", "netease"))
    if kind == "netease":
        return NeteaseDirect()
    if not url:
        raise RuntimeError(f"DSAPI_SOURCE={kind} needs DSAPI_SOURCE_URL")
    if kind == "ncm-api":
        return NcmApi(url, os.environ.get("DSAPI_NCM_COOKIE", ""))
    if kind == "meting":
        return Meting(url, os.environ.get("DSAPI_METING_SERVER", "netease"))
    raise RuntimeError(f"unknown DSAPI_SOURCE={kind}")
