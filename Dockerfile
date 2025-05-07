###############################
# Dockerfile for Hugging Face Spaces (Docker)
# Using Python 3.10-slim
###############################

# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY . .

# Expose port 80 for HF Spaces
EXPOSE 80

# Use Gunicorn with 2 workers binding to port 80
CMD ["gunicorn", "faceswap:app", "--bind=0.0.0.0:80", "--workers=2"]
