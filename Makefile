.PHONY: install lint format test clean

install:
	pip install -e ".[dev,sources,cex]"

lint:
	ruff check .
	mypy qmind/ --ignore-missing-imports

format:
	ruff format .

test:
	pytest tests/ -v --cov=qmind --cov-report=term-missing

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.PRECIOUS:
