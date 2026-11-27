PYTHON ?= .\venv\Scripts\python.exe

.PHONY: prepare features eda train monitor evaluate serve test lint clean

prepare:
	$(PYTHON) main.py prepare

features:
	$(PYTHON) main.py features

eda:
	$(PYTHON) main.py eda

train:
	$(PYTHON) main.py train --task all

monitor:
	$(PYTHON) main.py monitor

evaluate:
	$(PYTHON) main.py evaluate

all:
	$(PYTHON) main.py all

serve:
	$(PYTHON) main.py serve

test:
	.\venv\Scripts\pytest.exe

lint:
	.\venv\Scripts\ruff.exe check src scripts tests main.py
