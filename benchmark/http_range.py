"""Seekable read-only file over HTTP Range requests.

The sandbox proxy is honoured by ``requests`` (which reads the standard
``HTTPS_PROXY`` env vars) but not by ``aiohttp``/``urllib`` in this
environment, so random-access reads of large remote files (a Parquet file
or a ZIP central directory) go through ``requests`` here.

Only the byte ranges actually read are transferred: opening a 526 MB
Parquet file and reading its metadata columns costs a few MB, and listing
a ZIP central directory costs only its tail.
"""

from __future__ import annotations

import io

import requests


class HttpRangeFile(io.RawIOBase):
    """A seekable binary file backed by HTTP ``Range`` requests.

    ``url`` may redirect (e.g. huggingface.co -> CDN); redirects are
    resolved once in ``__init__`` and all later Range requests go to the
    final URL. Servers that answer out-of-range requests with 416 are
    handled by clamping reads to the known size (from ``HEAD``).
    """

    def __init__(self, url: str, session: requests.Session | None = None,
                 timeout: int = 60) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        resp = self.session.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        resp.close()
        self.url = resp.url
        head = self.session.head(self.url, timeout=timeout)
        length = head.headers.get("Content-Length")
        self._size = int(length) if length else None
        self._pos = 0

    @property
    def size(self) -> int | None:
        return self._size

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = (self._size or 0) + offset
        else:  # pragma: no cover - invalid whence
            raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def readinto(self, b: bytearray | memoryview) -> int:
        n = len(b)
        if n == 0:
            return 0
        if self._size is not None and self._pos >= self._size:
            return 0
        end = self._pos + n - 1
        if self._size is not None:
            end = min(end, self._size - 1)
        # Fill the buffer, re-requesting the remainder: a Range GET can
        # legitimately return fewer bytes than asked for (proxy / close).
        got = 0
        while got < n:
            if self._size is not None and self._pos + got >= self._size:
                break
            resp = self.session.get(
                self.url,
                headers={"Range": f"bytes={self._pos + got}-{end}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.content
            if not data:
                break
            b[got: got + len(data)] = data
            got += len(data)
        self._pos += got
        return got
