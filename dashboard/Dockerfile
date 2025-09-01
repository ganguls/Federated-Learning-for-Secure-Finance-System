FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p templates logs

# Copy dashboard code
COPY app.py .
COPY start_dashboard.py .
COPY templates/ templates/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 5000

# Run the dashboard
CMD ["python", "app.py"]
