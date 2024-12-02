# Credit-Card-Fraud-Detection-Program

A machine learning-powered program that detects fraudulent credit card transactions using ensemble methods and deep learning, built with FastAPI and Python.
 
## Important
- The dataset file is empty, that is because the initial creditcard.csv file is too large to be uploaded to Github. Here is the Kaggle link I used for the dataset.
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Features

- Real-time fraud detection predictions using XGBoost and Python
- Automated model training pipeline
- RESTful API endpoints for predictions
- Automated explanation generation using Hugging Face Transformers
- Model performance monitoring and health checks
- Standardized data preprocessing

## Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **XGBoost**: Gradient boosting framework for fraud detection
- **PyTorch**: Deep learning framework for neural network implementation
- **Hugging Face Transformers**: NLP capabilities for automated explanations
- **Scikit-learn**: Data preprocessing and model evaluation
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Seaborn & Matplotlib**: Data visualization
- **[Optional] Docker**: Containerized program for simple deployment

## Installation

1. Clone the repository
2. Install dependencies
3. Train the models (run the train_models.py file)
4. Start the FastAPI server (uvicorn main:app --reload)
5. The API will be available at `http://localhost:8000`
