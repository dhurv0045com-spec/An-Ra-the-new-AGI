import io
import sys


def _capture(fn, *args, **kwargs):
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_print_anra_dashboard_no_crash():
    from scripts.session_dashboard import print_anra_dashboard

    out = _capture(print_anra_dashboard, session_n=5, offline_minutes=90)
    assert "SESSION 5" in out
    assert "OFFLINE" in out
    assert "SYSTEM" in out


def test_dashboard_shows_flash_status():
    from scripts.session_dashboard import print_anra_dashboard

    out = _capture(print_anra_dashboard)
    assert "Flash Attention" in out


def test_dashboard_shows_benchmark_result():
    from scripts.session_dashboard import print_anra_dashboard
    from training.benchmark import BenchmarkResult

    result = BenchmarkResult(
        val_perplexity=24.3,
        rlvr_pass_at_1=0.58,
        civ_score=0.943,
        coherence=0.72,
    )
    out = _capture(print_anra_dashboard, benchmark_result=result)
    assert "LAST BENCHMARKS" in out
    assert "Validation perplexity" in out
    assert "CIV score" in out

