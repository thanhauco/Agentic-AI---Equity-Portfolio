# Production Dockerfile for AlphaAgents
FROM python:3.9-slim

# Install system dependencies for TA-Lib and other ML libs
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for logs and data
RUN mkdir -p .logs/governance .data

# Expose ports for Dashboard (8501) and API (8000)
EXPOSE 8501 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default to running the API gateway
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
