.PHONY: install install-ml test test-fast lint typecheck train-tiny serve verify clean help

# Default Python. Override with: make test PYTHON=python3.11
PYTHON ?= python

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dev dependencies
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev]"

install-ml:  ## Install full ML dependencies (sentence-transformers, faiss, etc.)
	pip install -e ".[dev,ml]"

test:  ## Run full test suite (skips GPU and Drive tests)
	$(PYTHON) -m pytest tests/ -m "not gpu" \
	  --ignore=tests/test_drive_session_manager_integration.py \
	  --ignore=tests/test_v2_drive_artifacts.py \
	  -q --tb=short

test-fast:  ## Run unit tests only (no I/O, completes in <30 seconds)
	$(PYTHON) -m pytest tests/test_anra_brain_unit.py \
	  tests/test_anra_package.py tests/test_schemas.py \
	  tests/test_bm25_memory.py tests/test_session_store.py \
	  tests/test_ci_health.py tests/test_hal.py tests/test_system_registry.py \
	  -q --tb=short

lint:  ## Run ruff linter
	$(PYTHON) -m ruff check anra/ tests/

typecheck:  ## Run mypy strict type checking on anra/ package
	$(PYTHON) -m mypy anra/ --strict --ignore-missing-imports

train-tiny:  ## Train for 100 steps on CPU with tiny config (smoke test)
	$(PYTHON) -m scripts.train --config config/tiny.yaml \
	  --max_steps 100 --device cpu

serve:  ## Start the AN-RA API server (development mode)
	PYTHONPATH=. uvicorn app:app --reload --port 8000

verify:  ## Verify project structure and imports
	$(PYTHON) -m scripts.verify_structure

clean:  ## Remove build artifacts and cache files
	find . -type d -name "__pycache__" | xargs rm -rf
	find . -type d -name "*.egg-info" | xargs rm -rf
	find . -type d -name ".pytest_cache" | xargs rm -rf
	find . -type d -name ".mypy_cache" | xargs rm -rf
	rm -rf dist/ build/
