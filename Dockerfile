FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire workspace into the container
COPY . .

# Expose port 5000 (standard Flask port)
EXPOSE 5000

# Run the production Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "dashboard.app:app"]
