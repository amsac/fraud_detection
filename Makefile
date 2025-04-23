.PHONY: setup test train build run-app docker-deploy

setup:
	@echo "==============================================="
	@echo "🚀  Setting up the project environment... 🚀"
	@echo "==============================================="
	@echo ""
	@echo "🔄 Checking if virtual environment exists..."
	[ -d "venv" ] || python3 -m venv venv
	@echo "✅ Virtual environment is ready!"
	@echo ""
	@echo "📦 Installing dependencies..."
	. venv/bin/activate && pip install -r requirements/requirements.txt
	. venv/bin/activate && pip install -r requirements/test_requirements.txt
	. venv/bin/activate && pip install -e .
	@echo "✅ Dependencies installed!"
	@echo ""
	@echo "==============================================="
	@echo "🎉 SETUP COMPLETE! 🎉"
	@echo "Run the following command to activate the venv:"
	@echo "➡️  source venv/bin/activate"
	@echo "==============================================="

test:
	@echo "🚀 Running tests...🚀"
	. venv/bin/activate && pytest

train:
	@echo "🚀 Training model...🚀"
	. venv/bin/activate && python fraud_transaction_detection/train_pipeline.py
	@echo "🎉 Trained model is saved under fraud_transaction_detection/trained_models! 🎉"

build:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ Virtual environment not detected! Please activate your venv first. 🛑"; \
		exit 1; \
	fi
	# @echo "🔄 Upgrading pip..."
	# pip install --upgrade pip
	@echo "📦 Installing build package..."
	pip install build
	@echo "🚀 Building the package..."
	python -m build
	@echo "🔍 Checking for the generated .whl file..."
	@if [ -f dist/*.whl ]; then \
		echo "✅ Found .whl file! Copying to fraud_transaction_detection_api/ 📂"; \
		cp dist/*.whl fraud_transaction_detection_api/; \
	else \
		echo "❌ No .whl file found in dist/. Build might have failed. 🛑"; \
		exit 1; \
	fi
	@echo "🎉 Build process completed successfully! 🚀"


run-app:
	@echo "🚀 Starting the fraud transaction detection API... 🔄"
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ Virtual environment not detected! Please activate your venv first. 🛑"; \
		exit 1; \
	fi
	@echo "📦 Installing dependencies..."
	cd fraud_transaction_detection_api && pip install -r requirements.txt && python -m app.main
	@echo "✅ Fraud transaction detection API is running! 🎉"


docker-deploy:
	@echo "🔍 Checking if Docker is running..."
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker is not running. Please start Docker first. 🛑"; exit 1; }
	@echo "✅ Docker is running! 🚀"
	@echo "📦 Building Docker image for Fraud transaction detection API..."
	docker build -t fraud_transaction_detection_api_image fraud_transaction_detection_api/
	@echo "🎉 Docker image built successfully!"
	@echo "🚢 Running the Docker container..."
	docker run -d --name fraud_detection_container -p 8000:8000 fraud_transaction_detection_api_image
	@echo "✅ Deployment successful! 🎯 Your API is running at http://localhost:8000 🚀"
