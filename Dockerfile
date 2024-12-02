# Use python runtime as base image
FROM python:3.10-slim
# Set working directory
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements.txt
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy rest of app
COPY ./app ./app
COPY ./dataset ./dataset
COPY ./app/models/saved_models ./app/models/saved_models
# Expose port 8000 to host 
EXPOSE 8000
# Run app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
