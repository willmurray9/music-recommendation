PYTHON_BOOTSTRAP ?= python3.11
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python

.PHONY: venv install install-dev test test-pytest pipeline-help web-install web-dev

venv:
	$(PYTHON_BOOTSTRAP) -m venv $(VENV)
	$(PYTHON) -m ensurepip --upgrade

install:
	$(PYTHON) -m ensurepip --upgrade
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m ensurepip --upgrade
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m unittest discover -s tests -q

test-pytest:
	$(PYTHON) -m pytest -q

pipeline-help:
	$(PYTHON) -m src.recommender_v2 --help

web-install:
	npm --prefix web install

web-dev:
	npm --prefix web run dev
