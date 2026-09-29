.PHONY: install lint format test cov clean pipeline dashboard

install:
	pip install -e ".[ml,dashboard,dev]"
	pre-commit install

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	ruff check --fix src tests

test:
	pytest

cov:
	pytest --cov=riverguard --cov-report=html

pipeline:
	python -m riverguard.pipeline --config configs/default.yaml --source data/raw/sample.mp4

dashboard:
	python -m riverguard.dashboard.app --config configs/default.yaml

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
