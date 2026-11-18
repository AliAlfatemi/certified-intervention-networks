PYTHON ?= python3

.PHONY: install test smoke build quick

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/run_smoke_test.py

build:
	$(PYTHON) -m build

quick:
	MPLBACKEND=Agg $(PYTHON) run_all.py --quick --seeds 1 --output outputs/quick --no-paper-figures --jobs 1 --require-empty-output
