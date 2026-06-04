import asyncio
import pytest
import tempfile
from pathlib import Path
from app import SQLiteSessionStore

@pytest.fixture
def store(tmp_path):
    s = SQLiteSessionStore(tmp_path / "sessions.db", max_history=3)
    asyncio.get_event_loop().run_until_complete(s.initialize())
    return s

def test_write_and_read_history(store):
    asyncio.get_event_loop().run_until_complete(
        store.save_history("sess1", [{"role": "user", "content": "hi"}])
    )
    history = asyncio.get_event_loop().run_until_complete(store.get_history("sess1"))
    assert len(history) == 1
    assert history[0]["content"] == "hi"

def test_history_trimmed_to_max(store):
    msgs = [{"role": "user", "content": str(i)} for i in range(10)]
    asyncio.get_event_loop().run_until_complete(store.save_history("sess2", msgs))
    history = asyncio.get_event_loop().run_until_complete(store.get_history("sess2"))
    assert len(history) == 3  # max_history=3

def test_rate_limit_blocks_after_threshold(store):
    loop = asyncio.get_event_loop()
    for _ in range(30):
        allowed = loop.run_until_complete(store.check_rate_limit("1.2.3.4", window_seconds=60, max_requests=30))
        assert allowed
    blocked = loop.run_until_complete(store.check_rate_limit("1.2.3.4", window_seconds=60, max_requests=30))
    assert not blocked

def test_survives_restart(tmp_path):
    loop = asyncio.get_event_loop()
    store1 = SQLiteSessionStore(tmp_path / "s.db")
    loop.run_until_complete(store1.initialize())
    loop.run_until_complete(store1.save_history("abc", [{"role": "user", "content": "hello"}]))
    store2 = SQLiteSessionStore(tmp_path / "s.db")
    loop.run_until_complete(store2.initialize())
    history = loop.run_until_complete(store2.get_history("abc"))
    assert history[0]["content"] == "hello"
