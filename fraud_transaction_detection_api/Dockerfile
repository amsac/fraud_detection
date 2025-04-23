# Use an official lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements first to leverage Docker caching
COPY requirements.txt .



# Copy the model package and the application files
COPY fraud_transaction_detection_model-0.0.1-py3-none-any.whl .
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Install the model package
RUN pip install fraud_transaction_detection_model-0.0.1-py3-none-any.whl

# Expose the FastAPI default port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
