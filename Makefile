# Makefile for Fake News Detection Project

.PHONY: help install test train api demo docker clean lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run all tests"
	@echo "  test-unit   - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  train       - Train models"
	@echo "  train-fast  - Train models (skip DeBERTa)"
	@echo "  api         - Start FastAPI server"
	@echo "  demo        - Start Streamlit demo"
	@echo "  docker      - Build and run Docker container"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run Docker container"
	@echo "  clean       - Clean temporary files"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black"
	@echo "  extract-data - Extract zipped datasets"

# Installation
install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Extract datasets
extract-data:
	@echo "Extracting datasets..."
	python -c "import zipfile; import os; \
	for f in ['Fake.csv.zip', 'True.csv.zip', 'fake_or_real_news.csv.zip']: \
		if os.path.exists(f'data/raw/{f}'): \
			zipfile.ZipFile(f'data/raw/{f}').extractall('data/raw/')"

# Testing
test: test-unit test-integration

test-unit:
	python -m pytest tests/unit/ -v --cov=src --cov-report=html

test-integration:
	python -m pytest tests/integration/ -v

# Training
train: extract-data
	python train_pipeline.py

train-fast: extract-data
	python train_pipeline.py --skip-deberta

# API and Demo
api:
	python -m uvicorn src.deployment.api:app --reload --host 0.0.0.0 --port 8000

demo:
	streamlit run web_demo.py --server.port 8501

# Docker
docker: docker-build docker-run

docker-build:
	docker build -t fake-news-detection .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/logs:/app/logs fake-news-detection

docker-compose-up:
	docker-compose up --build

docker-compose-down:
	docker-compose down

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	python -m pylint src/ --disable=C0114,C0115,C0116

format:
	black src/ tests/ --line-length=100
	isort src/ tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

# Development setup
dev-setup: install extract-data
	@echo "Setting up development environment..."
	mkdir -p logs results models/checkpoints
	@echo "Development setup complete!"

# Production deployment
deploy-prod:
	@echo "Deploying to production..."
	docker-compose -f docker-compose.prod.yml up -d
	@echo "Production deployment complete!"

# Quick demo with sample data
quick-demo:
	python demo_usage.py

# Run all quality checks
check-all: lint test
	@echo "All quality checks passed!"

# Performance benchmark
benchmark:
	python -c "from src.pipeline import FakeNewsDetector; \
	import time; \
	detector = FakeNewsDetector(); \
	detector.load_datasets(); \
	detector.train_traditional_models(); \
	texts = ['Sample text'] * 100; \
	start = time.time(); \
	results = detector.predict(texts, model_type='tfidf'); \
	end = time.time(); \
	print(f'Processed 100 texts in {end-start:.2f}s ({(end-start)*10:.1f}ms per text)')"