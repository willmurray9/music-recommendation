PYTHON_BOOTSTRAP ?= python3.11
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
CONFIG ?= config/recommender_v2.dev.toml
RUN_ID ?= dev-verify

.PHONY: venv install install-dev test test-pytest pipeline-help prepare-model-data train-model verify-model web-install web-dev

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

prepare-model-data:
	$(PYTHON) -m src.recommender_v2 --config $(CONFIG) --run-id $(RUN_ID) collect_spotify
	$(PYTHON) -m src.recommender_v2 --config $(CONFIG) --run-id $(RUN_ID) build_corpus
	$(PYTHON) -m src.recommender_v2 --config $(CONFIG) --run-id $(RUN_ID) enrich_metadata
	$(PYTHON) -m src.recommender_v2 --config $(CONFIG) --run-id $(RUN_ID) split_eval

train-model:
	$(PYTHON) -m src.recommender_v2 --config $(CONFIG) --run-id $(RUN_ID) train_retrieval
	$(PYTHON) -m src.recommender_v2 --config $(CONFIG) --run-id $(RUN_ID) train_reranker
	$(PYTHON) -m src.recommender_v2 --config $(CONFIG) --run-id $(RUN_ID) evaluate

verify-model: prepare-model-data train-model

web-install:
	npm --prefix web install

web-dev:
	npm --prefix web run dev
