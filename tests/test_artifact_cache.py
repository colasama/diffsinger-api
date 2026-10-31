import json

import pytest

from dsapi import cache, settings


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'WORK', tmp_path)
    monkeypatch.setattr(settings, 'CACHE_ENABLED', True)
    monkeypatch.setattr(settings, 'CACHE_TTL', 86400)
    monkeypatch.setattr(settings, 'CACHE_MAX_BYTES', 100)
    source = tmp_path / 'source'
    source.write_bytes(b'original')
    return cache.ArtifactCache('alice'), source


def test_restored_files_are_independent_and_owners_are_isolated(artifacts, tmp_path):
    c, source = artifacts
    key = c.key('f0', {'model': 'one'})
    c.store(key, {'f0.npy': source})
    source.unlink()
    target = tmp_path / 'another/job/f0.npy'
    assert not cache.ArtifactCache('bob').restore(key, {'f0.npy': target})
    assert c.restore(key, {'f0.npy': target})
    target.write_bytes(b'changed')
    assert c.restore(key, {'f0.npy': target})
    assert target.read_bytes() == b'original'
    assert c.key('f0', {'model': 'two'}) != key
    cache.purge_owner('alice')
    assert not c.restore(key, {'f0.npy': target})


def test_corruption_expiry_and_disabled_cache_are_misses(artifacts, tmp_path, monkeypatch):
    c, source = artifacts
    key = c.key('f0', {})
    c.store(key, {'f0.npy': source})
    target = tmp_path / 'target'
    target.write_bytes(b'untouched')
    (c.root / key / 'f0.npy').write_bytes(b'corrupt')
    assert not c.restore(key, {'f0.npy': target})
    assert target.read_bytes() == b'untouched'
    c.store(key, {'f0.npy': source})
    manifest = c.root / key / 'manifest.json'
    meta = json.loads(manifest.read_text()); meta['created'] = 0
    manifest.write_text(json.dumps(meta))
    assert not c.restore(key, {'f0.npy': target})
    monkeypatch.setattr(settings, 'CACHE_ENABLED', False)
    c.store('disabled', {'f0.npy': source})
    assert not (c.root / 'disabled').exists()


def test_capacity_is_global_and_model_replacement_invalidates_fingerprint(artifacts, tmp_path, monkeypatch):
    c, source = artifacts
    monkeypatch.setattr(settings, 'CACHE_MAX_BYTES', 10)
    before = cache.files([source])
    c.store('first', {'f0.npy': source})
    cache.ArtifactCache('bob').store('second', {'f0.npy': source})
    assert not (c.root / 'first').exists()
    source.write_bytes(b'new-data')
    assert before != cache.files([source])
    assert cache.digest(source) != cache.digest(cache.owner_root('bob') / 'second/f0.npy')
