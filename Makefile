PYTHON ?= python3
export PYTHONPATH := src

.PHONY: test doctor demo lint sodh audit baselines smoke

test:
	$(PYTHON) -m pytest -q

doctor:
	$(PYTHON) scripts/doctor.py

demo:
	$(PYTHON) scripts/dry_run.py

lint:
	$(PYTHON) -m ruff check src tests scripts

sodh:
	$(PYTHON) scripts/generate_sodh.py

audit:
	$(PYTHON) scripts/audit_sodh.py

baselines:
	$(PYTHON) scripts/run_symbolic_baselines.py

smoke:
	PYTHON=$(PYTHON) bash scripts/smoke_cpu.sh
