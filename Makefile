# MAIF Production Makefile

.PHONY: help build test lint format clean install docker-build docker-run deploy docs

# Default target
help:
	@echo "MAIF Production Commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make build        - Build the package"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make deploy       - Deploy to AWS"
	@echo "  make docs         - Generate documentation"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

install-prod:
	pip install -e ".[production]"

# Building
build: clean
	python setup.py sdist bdist_wheel

# Testing
test:
	pytest tests/ -v --cov=maif --cov-report=term-missing

test-unit:
	pytest tests/ -v -m "not integration and not aws" --cov=maif

test-integration:
	pytest tests/ -v -m "integration" --cov=maif

test-aws:
	pytest tests/ -v -m "aws" --cov=maif

# Code quality
lint:
	flake8 maif/ --max-line-length=120
	mypy maif/ --ignore-missing-imports
	black --check maif/

format:
	black maif/
	isort maif/

security-check:
	bandit -r maif/ -ll
	safety check

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.maif" -delete

# Docker operations
docker-build:
	docker build -t maif:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f maif

# AWS deployment
deploy-lambda:
	python -m maif.aws_deployment deploy-lambda \
		--function-name maif-production \
		--handler maif.handler.lambda_handler \
		--runtime python3.11

deploy-ecs:
	python -m maif.aws_deployment deploy-ecs \
		--cluster maif-cluster \
		--service maif-service \
		--image maif:latest

deploy-cloudformation:
	aws cloudformation deploy \
		--template-file cloudformation/maif-infrastructure.yaml \
		--stack-name maif-production \
		--capabilities CAPABILITY_IAM

# Documentation
docs:
	cd docs && npm run docs:build

docs-serve:
	cd docs && npm run docs:dev

# Performance testing
benchmark:
	python benchmarks/maif_benchmark_suite.py

benchmark-aws:
	python benchmarks/bedrock_swarm_benchmark.py

# Monitoring
monitor-start:
	docker-compose up -d prometheus grafana

monitor-stop:
	docker-compose stop prometheus grafana

# Local development
dev-setup: install-dev
	pre-commit install
	cp .env.example .env

run-local:
	python -m maif.cli serve --debug

# Production checks
prod-check:
	python -m maif.config validate_production_config
	python -m maif.health_check --full

# Release
release-patch:
	bumpversion patch
	git push --tags

release-minor:
	bumpversion minor
	git push --tags

release-major:
	bumpversion major
	git push --tags