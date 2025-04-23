# 🚦 AI-Powered Fraud Detection System

This project implements a robust fraud detection system for banking transactions using machine learning. It leverages AWS services for deployment, Optuna for model optimization, and FastAPI for API services.

---

## 🔧 Tech Stack

- **FastAPI** for API services
- **Random Forest Regressor** for transaction classification
- **Optuna** for hyperparameter tuning
- **GitHub Actions** for CI/CD automation
- **Docker, AWS ECR, ECS, Lambda, and EventBridge** for containerization and deployment

---

## 🔄 CI/CD Pipeline Overview

1. GitHub Actions pipeline is triggered on code push.
2. Model is trained using the latest dataset.
3. Optuna performs model tuning.
4. Trained model is saved locally.
5. Docker image is built and pushed to Amazon ECR.
6. AWS EventBridge detects new image and triggers a Lambda function.
7. Lambda updates the ECS service with the new Docker image.

---

## 🛠️ Setup Instructions

### 📁 Clone the Repo and Setup Environment

```bash
make setup
```

- Initializes a virtual environment (`venv`)
- Installs required and test dependencies
- Installs the project in editable mode

### ▶️ Activate Virtual Environment

```bash
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

---

## 🧪 Running Tests

```bash
make test
```

Runs test suite using `pytest`.

---

## 🎓 Train the Model

```bash
make train
```

Trains the model using the current best hyperparameters and stores it in `fraud_transaction_detection/trained_models/`.

---

## 📦 Build the Package

```bash
make build
```

- Builds the Python wheel using `python -m build`
- Copies the `.whl` to the FastAPI app directory

---

## 🚀 Run the Application

```bash
make run-app
```

Runs the FastAPI server locally at:

```
http://0.0.0.0:8001/
```

---

## 🐳 Docker Deployment

```bash
make docker-deploy
```

- Builds Docker image from API source
- Runs the Docker container locally
- Ensures the built `.whl` is available in the container context

---

## ⚠️ Requirements

- Python 3.x
- Docker installed and running
- Proper AWS credentials and GCP credential file setup for accessing datasets and deploying services

---

## 📘 Notes

- Always activate the virtual environment before running project scripts
- Ensure AWS and Docker are correctly configured

---

## 🧠 Contributions

Feel free to submit issues, pull requests, or feedback to improve the project.

---

## 📜 License

MIT License