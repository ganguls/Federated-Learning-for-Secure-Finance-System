FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create client directory
RUN mkdir -p /app/client

# Expose port (clients don't need external ports but good practice)
EXPOSE 8081

# Default command (will be overridden by docker-compose)
CMD ["python", "client.py"]
