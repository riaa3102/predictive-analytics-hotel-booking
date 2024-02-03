# Define directories
MLEXPERIMENTS_DIR	:= experiments\mlflow
MLRUNS_DIR			:= $(MLEXPERIMENTS_DIR)\mlruns
MLARTIFACTS_DIR		:= $(MLEXPERIMENTS_DIR)\mlartifacts

.PHONY: install load preprocess data mlflow baseline train clean help

SHELL = /bin/bash
.SHELLFLAGS = -ec

install:
	@echo "Installing Dependencies..."
	@pip install -r requirements.txt

# Download data
load:
	@echo "Downloading Data..."
	@python -m src.data.load

# Run Exploratory Data Analysis notebook
eda:
	@echo "Running EDA notebook..."
	@jupyter-notebook notebooks/"Exploratory Data Analysis (EDA)".ipynb

# Preprocess data (depends on 'load')
preprocess: load
	@echo "Processing Data..."
	@python -m src.data.preprocess

# Complete data processing pipeline
data: preprocess

# Start MLflow UI and Tracking Server
mlflow:
	@start /B mlflow ui --backend-store-uri $(MLRUNS_DIR) --default-artifact-root $(MLARTIFACTS_DIR) --host 127.0.0.1 --port 8000 &

# Run baseline model (depends on 'data' and 'mlflow')
baseline: data mlflow
	@echo "Running Baseline Model..."
	@python -m src.models.baseline_model

# Train ML model (depends on 'data' and 'mlflow')
train: data mlflow
	@echo "Training ML Model..."
	@python -m src.models.train_model --model=$(model)
	@echo "Running MLflow Search..."
	@python -m src.experiments.mlflow_search

# Train all ML models (depends on 'data' and 'mlflow')
train_all: data mlflow
	@echo "Training Logistic Regression Model..."
	@python -m src.models.train_model --model=lr
	@echo "Training Decision Tree Model..."
	@python -m src.models.train_model --model=dt
	@echo "Training Random Forest Model..."
	@python -m src.models.train_model --model=rf
	@echo "Training XGBoost Model..."
	@python -m src.models.train_model --model=xgboost
	@echo "Training CatBoost Model..."
	@python -m src.models.train_model --model=catboost
	@echo "Training MLP Model..."
	@python -m src.models.train_model --model=mlp
	@echo "Running MLflow Search..."
	@python -m src.experiments.mlflow_search

app_input:
	@echo "Creating App input example..."
	@python -m app_input_example.create_app_input_example

app:
	@echo "Running App..."
	@uvicorn app:app --host=127.0.0.1 --port=5000

hotel_app:
	@echo "Running App..."
	@uvicorn app:app --host=0.0.0.0 --port=5000

clean:
	@echo "Deleting Files..."
	@if exist data (rmdir /s /q data)
	@if exist logs (rmdir /s /q logs)
	@if exist experiments (rmdir /s /q experiments)
	# @if exist $(MLRUNS_DIR) (for /d %%i in ($(MLRUNS_DIR)\*) do @rmdir /s /q %%i) else (true)
	# @if exist $(MLARTIFACTS_DIR) (for /d %%i in ($(MLARTIFACTS_DIR)\*) do @rmdir /s /q %%i) else (true)

help:
	@echo "Available targets:"
	@echo "  make install           		: Install dependencies"
	@echo "  make load              		: Download the data"
	@echo "  make eda               		: Run exploratory data analysis"
	@echo "  make preprocess        		: Preprocess the data"
	@echo "  make data              		: Complete data processing pipeline ('load' and 'preprocess')"
	@echo "  make mlflow            		: Start MLflow UI and Tracking Server"
	@echo "  make baseline          		: Run baseline model"
	@echo "  make train model=<model-name>	: Train ML model"
	@echo "  make train_all          		: Train all ML models"
	@echo "  make app          				: Run FastAPI app"
	@echo "  make clean             		: Remove generated files and directories"

