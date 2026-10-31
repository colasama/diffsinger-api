"""Bounded, per-owner artifact cache. Jobs always receive independent copies."""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from threading import RLock

from . import settings

_lock = RLock()
_log = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _entry_lock(path: str):
    return RLock()


@lru_cache(maxsize=512)
def _digest(path: str, size: int, mtime: int, ctime: int) -> str:
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(path: Path) -> str:
    path = path.resolve()
    st = path.stat()
    return _digest(str(path), st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def files(paths, *, content=False):
    """Fingerprint model installations cheaply; hash input/recipe contents."""
    result = []
    for path in sorted(set(map(Path, paths))):
        if path.is_dir():
            result.extend(files((p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts), content=content))
        elif path.is_file():
            st = path.stat()
            result.append((str(path.resolve()), digest(path) if content else (st.st_size, st.st_mtime_ns, st.st_ctime_ns)))
        else:
            result.append((str(path.resolve()), None))
    return result


def owner_root(owner: str | None) -> Path:
    ident = hashlib.sha256((owner or '__legacy__').encode()).hexdigest()
    return settings.WORK / '.cache' / ident


def purge_owner(owner: str | None) -> None:
    with _lock:
        shutil.rmtree(owner_root(owner), ignore_errors=True)


class ArtifactCache:
    def __init__(self, owner: str | None):
        self.root = owner_root(owner)

    def guard(self, key: str):
        """Coalesce simultaneous downloads for the same owner and source."""
        return _entry_lock(str(self.root / key))

    def key(self, stage: str, parameters: dict) -> str:
        recipe = files(Path(__file__).parent.glob('*.py'), content=True)
        recipe += files((Path(__file__).parent / 'runners').glob('*.py'), content=True)
        raw = json.dumps([1, stage, recipe, parameters], sort_keys=True, default=str).encode()
        return stage + '-' + hashlib.sha256(raw).hexdigest()

    def restore(self, key: str, outputs: dict[str, Path], *, ttl: float | None = None) -> bool:
        if not settings.CACHE_ENABLED:
            return False
        entry = self.root / key
        with _lock:
            try:
                manifest = json.loads((entry / 'manifest.json').read_text())
                if time.time() - manifest['created'] > (settings.CACHE_TTL if ttl is None else ttl):
                    return False
                if set(manifest['files']) != set(outputs):
                    return False
                # Verify the whole entry before touching any job output.
                for name, expected in manifest['files'].items():
                    if digest(entry / name) != expected:
                        return False
                for name, target in outputs.items():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(entry / name, target)
                os.utime(entry, None)
                return True
            except (OSError, ValueError, KeyError, TypeError):
                return False

    def store(self, key: str, outputs: dict[str, Path]) -> None:
        if not settings.CACHE_ENABLED:
            return
        with _lock:
            try:
                size = sum(p.stat().st_size for p in outputs.values())
                if size > settings.CACHE_MAX_BYTES:
                    return
                self.root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='.pending-', dir=self.root) as tmp:
                    staging = Path(tmp) / 'entry'
                    staging.mkdir()
                    hashes = {}
                    for name, source in outputs.items():
                        if Path(name).name != name:
                            raise ValueError('Cache artifact names must be basenames')
                        shutil.copyfile(source, staging / name)
                        hashes[name] = digest(staging / name)
                    (staging / 'manifest.json').write_text(json.dumps({
                        'created': time.time(), 'bytes': size, 'files': hashes}))
                    entry = self.root / key
                    shutil.rmtree(entry, ignore_errors=True)
                    staging.rename(entry)
                self._prune()
            except OSError:
                _log.warning('Could not store %s in artifact cache', key, exc_info=True)

    def _prune(self):
        entries = []
        # The capacity applies across users, not once per user.
        for manifest in (settings.WORK / '.cache').glob('*/*/manifest.json'):
            try:
                meta = json.loads(manifest.read_text())
                if time.time() - meta['created'] > settings.CACHE_TTL:
                    shutil.rmtree(manifest.parent)
                else:
                    entries.append((manifest.parent.stat().st_mtime, manifest.parent, meta['bytes']))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        remaining = sum(size for _, _, size in entries)
        for _, path, size in sorted(entries):
            if remaining <= settings.CACHE_MAX_BYTES:
                break
            shutil.rmtree(path)
            remaining -= size
