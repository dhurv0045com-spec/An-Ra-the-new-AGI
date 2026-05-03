import pytest

from memory.memory_router import MemoryRouter


class _ESV:
    def memory_write_threshold(self):
        return 0.75


def test_memory_router_uses_esv_write_threshold(tmp_path):
    pytest.importorskip("numpy")
    router = MemoryRouter(dim=8, faiss_index_path=tmp_path / "episodic.faiss", esv=_ESV())
    result = router.write("low salience", metadata={"salience": 0.5})
    assert result.tier == "short_term"
    assert router.short_term[-1]["metadata"]["esv_threshold"] == 0.75


def test_memory_router_keeps_high_salience_episodic(tmp_path):
    pytest.importorskip("numpy")
    router = MemoryRouter(dim=8, faiss_index_path=tmp_path / "episodic.faiss", esv=_ESV())
    result = router.write("high salience", metadata={"salience": 0.95})
    assert result.tier == "episodic"
