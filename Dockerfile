# Federated Learning Attack Detection System
# ==========================================
# This Dockerfile builds a container for running federated learning
# with malicious client detection using LDP and K-means clustering.

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/results /app/dataset /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN useradd -m -u 1000 fluser && \
    chown -R fluser:fluser /app
USER fluser

# Expose port (if needed for web interface)
EXPOSE 5000

# Default command
CMD ["python", "scripts/run_attack_detection.py", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"



