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

# Copy server code
COPY server.py .
COPY enhanced_server_strategy.py .
COPY detection_adapter.py .

# Create necessary directories
RUN mkdir -p /app/detection_results /app/logs

# Expose port
EXPOSE 8080

# Run the enhanced server with detection
CMD ["python", "enhanced_server_strategy.py"]
