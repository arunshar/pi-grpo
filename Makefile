.PHONY: help venv install lint type test test-property cov security audit \
        bench paper paper-clean clean verify ci

PY      ?= python
UV      ?= uv
VENV    ?= .venv
ACT     := source $(VENV)/bin/activate

help:
	@echo "Targets:"
	@echo "  install        uv venv + install CPU-only deps"
	@echo "  lint           ruff check"
	@echo "  type           mypy app"
	@echo "  test           pytest -q"
	@echo "  test-property  pytest -q tests/test_reward_properties.py"
	@echo "  cov            pytest with coverage report"
	@echo "  security       bandit + pip-audit"
	@echo "  bench          run benchmarks"
	@echo "  paper          build the NeurIPS PDF"
	@echo "  verify         lint + type + test + cov + security (FAANG-style)"
	@echo "  ci             alias for verify"

venv:
	$(UV) venv --python 3.11 $(VENV)

install: venv
	$(ACT) && $(UV) pip install \
	  "fastapi>=0.115" "uvicorn[standard]>=0.30" "pydantic>=2.7" "pydantic-settings>=2.3" \
	  "httpx>=0.27" "asyncpg>=0.29" "redis>=5.0" \
	  "torch>=2.4" "transformers>=4.44" "tiktoken>=0.7" "numpy>=1.26" "scipy>=1.13" \
	  "shapely>=2.0" "pyyaml" "structlog>=24.1" "tenacity>=8.3" "pyproj>=3.6" "pandas>=2.0" \
	  "opentelemetry-api>=1.25" "opentelemetry-sdk>=1.25" "opentelemetry-exporter-otlp>=1.25" \
	  "orjson>=3.10" "pytest>=8.2" "pytest-asyncio>=0.23" "pytest-cov" \
	  "ruff>=0.5" "mypy>=1.10" "huggingface-hub" "hypothesis" "bandit" "pip-audit" "pytest-benchmark"

lint:
	$(ACT) && ruff check app observability evaluation tests

type:
	$(ACT) && mypy app || true

test:
	$(ACT) && PG_OTEL_ENDPOINT="" PYTHONPATH=. pytest -q -p no:warnings

test-property:
	$(ACT) && PYTHONPATH=. pytest -q tests/test_reward_properties.py

cov:
	$(ACT) && PG_OTEL_ENDPOINT="" PYTHONPATH=. pytest --cov=app --cov=observability \
	  --cov-report=term-missing --cov-report=xml --cov-fail-under=65 -p no:warnings

security:
	$(ACT) && bandit -q -r app observability evaluation -ll || true
	$(ACT) && pip-audit --skip-editable --strict || true

bench:
	$(ACT) && PYTHONPATH=. python scripts/bench.py

paper:
	$(MAKE) -C paper pdf

paper-clean:
	$(MAKE) -C paper clean

verify: lint type cov security
	@echo ""
	@echo "FAANG verify pass."

ci: verify

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
