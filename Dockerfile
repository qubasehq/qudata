# Multi-stage Dockerfile for QuData LLM Processing System
# Optimized for production deployment with security and performance considerations

# Build stage - Install dependencies and build the application
FROM python:3.11-slim AS builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata labels
LABEL maintainer="Qubase Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="QuData" \
      org.label-schema.description="LLM Data Processing System" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0"

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project metadata and source for editable install
COPY pyproject.toml ./
COPY requirements.txt* ./
COPY README.md ./
COPY src/ ./src/

# Install dependencies and package (editable)
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir gunicorn uvicorn[standard] && \
    pip cache purge

# Production stage - Create the final runtime image
FROM python:3.11-slim AS production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Database clients
    libpq5 \
    # Shell and networking utilities required by entrypoint
    bash \
    netcat-openbsd \
    # OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    libtesseract5 \
    # Image processing
    libopencv-dev \
    # XML processing
    libxml2 \
    libxslt1.1 \
    # Network tools
    curl \
    wget \
    # Process management
    supervisor \
    # Utilities
    jq \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r qudata && useradd -r -g qudata -d /app -s /bin/bash qudata

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=qudata:qudata . .

# Install QuData in development mode
RUN pip install --no-cache-dir -e .

# Create necessary directories with proper permissions
RUN mkdir -p \
    data/raw \
    data/staging \
    data/processed \
    data/exports \
    logs \
    tmp \
    configs/templates \
    && chown -R qudata:qudata /app \
    && chmod -R 755 /app \
    && chmod -R 777 /app/data \
    && chmod -R 777 /app/logs \
    && chmod -R 777 /app/tmp

# Copy configuration files
COPY --chown=qudata:qudata configs/ configs/

# Copy supervisor configuration
COPY --chown=qudata:qudata docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy entrypoint script
COPY --chown=qudata:qudata docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QUDATA_CONFIG_PATH=/app/configs \
    QUDATA_DATA_PATH=/app/data \
    QUDATA_LOG_PATH=/app/logs \
    QUDATA_TEMP_PATH=/app/tmp \
    # Performance settings
    WORKERS=4 \
    MAX_WORKERS=8 \
    BATCH_SIZE=100 \
    MAX_MEMORY=4GB \
    # Security settings
    QUDATA_SECRET_KEY="" \
    # Database settings
    DB_HOST=localhost \
    DB_PORT=5432 \
    DB_NAME=qudata \
    DB_USER=qudata \
    DB_PASSWORD="" \
    # Redis settings
    REDIS_URL=redis://localhost:6379/0 \
    # API settings
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    # Logging
    LOG_LEVEL=INFO

# Expose ports
EXPOSE 8000 8001 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER qudata

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command - can be overridden
CMD ["api"]

# Development stage - For development with hot reload
FROM production AS development

# Switch back to root to install dev dependencies
USER root

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    fitz \
    pre-commit \
    jupyter \
    ipython

# Install additional development system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Switch back to qudata user
USER qudata

# Override default command for development
CMD ["api", "--reload"]