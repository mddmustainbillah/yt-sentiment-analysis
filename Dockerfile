# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Configure pip with increased timeout and retries
RUN pip config set global.timeout 1000
RUN pip config set global.retries 10

# Copy requirements file
COPY flask_app/requirements.txt /app/requirements.txt

# Install Python dependencies with optimizations
# Split installation into chunks to avoid timeout issues
RUN pip install --no-cache-dir --timeout=1000 \
    boto3==1.35.36 \
    Flask==3.0.3 \
    Flask_Cors==5.0.0 \
    joblib==1.4.2

# Install scientific computing packages separately (these are the large ones)
RUN pip install --no-cache-dir --timeout=1000 \
    numpy==2.1.2

RUN pip install --no-cache-dir --timeout=1000 \
    pandas==2.2.3

RUN pip install --no-cache-dir --timeout=1000 \
    matplotlib==3.9.2

# Install ML packages
RUN pip install --no-cache-dir --timeout=1000 \
    lightgbm==4.5.0

RUN pip install --no-cache-dir --timeout=1000 \
    mlflow==2.17.0 \
    mlflow_skinny==2.17.0

# Install remaining packages
RUN pip install --no-cache-dir --timeout=1000 \
    nltk==3.9.1 \
    wordcloud==1.9.3

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy application code
COPY flask_app/ /app/

# Create MLflow directories
RUN mkdir -p /app/mlruns

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5005

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5005/health || exit 1

# Run the application
CMD ["python", "app.py"]