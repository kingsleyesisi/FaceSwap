###############################
# Dockerfile for Render.com
# Using Python 3.10
###############################

# Base image
FROM python:3.10-slim

# Set working directory
# WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy application code and models
COPY . .

# Expose the port Render uses via $PORT
ENV PORT=10000
EXPOSE 10000

# Production startup command using Gunicorn
CMD ["gunicorn", "faceswap:app", "--bind=0.0.0.0:10000", "--workers=2"]
