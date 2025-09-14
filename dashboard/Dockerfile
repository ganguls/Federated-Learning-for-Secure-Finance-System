FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional detection dependencies
RUN pip install --no-cache-dir torch scipy

# Create necessary directories
RUN mkdir -p templates logs demo_results detection_results

# Copy dashboard code
COPY app.py .
COPY start_dashboard.py .
COPY detection_api.py .
COPY templates/ templates/

# Copy research demonstration files (will be mounted as volumes)
# These files are mounted from the host system in docker-compose.yml

# Create logs and results directories
RUN mkdir -p logs demo_results detection_results

# Set environment variables for research demo and detection
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV DASHBOARD_PORT=5000
ENV DETECTION_ENABLED=true
ENV DETECTION_METHOD=kmeans
ENV LDP_EPSILON=1.0

# Expose port
EXPOSE 5000

# Run the dashboard
CMD ["python", "app.py"]
