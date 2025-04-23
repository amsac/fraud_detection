### 🛡️ Project Overview:  Fraud Detection System using AWS (ECR, ECS, Lambda) and DVC

**Fraud Transaction Prediction** is a machine learning project aimed at identifying fraudulent bank transactions using various input features. The project uses a supervised learning approach with a **Random Forest Regression model**, which achieved an accuracy of **99%** on the labeled dataset.

### 🧰 Tools & Technologies
- **Amazon ECR & ECS**: Docker image is built for a FastAPI application and pushed to **Amazon ECR**. Deployment is handled through **Amazon ECS**.
- **DVC (Data Version Control)**: Used for versioning datasets stored in **Google Cloud Storage**.
- **Optuna**: Employed for hyperparameter tuning to identify the best model configuration.
- **GitHub Actions**: CI/CD pipeline automates the workflow.

### ⚙️ CI/CD Pipeline Workflow
1. Pipeline is triggered in GitHub Actions.
2. Data is pulled from **Google Cloud Storage** using DVC.
3. **Optuna** performs hyperparameter tuning to find the best parameters.
4. The model is trained using these optimal parameters.
5. A **Docker image** is built for the FastAPI app and pushed to **Amazon ECR**.
6. An **Amazon EventBridge** rule (previously CloudWatch Events) detects the new image push.
7. This triggers an **AWS Lambda function**, which is configured with the ECS **cluster**, **service**, and **task definition** details.
8. The Lambda function deploys the latest image to the ECS service automatically.

---

## 🛠 Setup Instructions
To get started, clone the repository and run the following command to set up the environment:

```sh
make setup
```

This will:
- Create a virtual environment (`venv`) if it doesn’t exist.
- Install all required dependencies from `requirements/requirements.txt`.
- Install test dependencies from `requirements/test_requirements.txt`.
- Install the package in editable mode.

### 🏃 Activating Virtual Environment
After setup, activate the virtual environment manually:

```sh
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

## ✅ Running Tests
To run tests, use:

```sh
make test
```

This will execute all tests using `pytest`.

## 🎯 Training the Model
To train the model, run:

```sh
make train
```

This will execute the training pipeline and save the trained model under `fraud_transaction_detection/trained_models/`.

## 🏗️ **Building the Package**
To build the package, use:

```sh
make build
```

This command will:  
- Build the Python package using `python -m build`  
- Verify if the `.whl` file is generated and copy it to the `fraud_transaction_detection_api/` folder  

Ensure you have Python and `pip` installed before running this command.  

---

## 🚀 **Running the Application**
To start the fraud transaction detection API, run:

```sh
make run-app
```

This command will:  
- Start the application on http://0.0.0.0:8001/ 

Ensure the virtual environment (`venv`) is set up before running this command.  

---

## 🐳 **Deploying with Docker**
To build and run the application inside a Docker container, use:

```sh
make docker-deploy
```

This command will:  
- Check if Docker is running  
- Build a Docker image using the `Dockerfile` inside `fraud_transaction_detection_api/`  
- Run the container with the built image
  
Ensure the `.whl` file is generated and copy it to the `fraud_transaction_detection_api/` folder 
Ensure Docker is installed and running before executing this command.  

---



## 💡 Additional Information
- Ensure Python 3.x is installed before running `make setup`.
- If you encounter any issues, try running commands inside the virtual environment manually.

🚀 Happy coding!
