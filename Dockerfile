# Production Dockerfile for MAIF
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Production image
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r maif && useradd -r -g maif maif

# Set working directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Copy application code
COPY . .

# Install dependencies from wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && \
    pip install --no-cache-dir -e . && \
    rm -rf /wheels

# Set ownership
RUN chown -R maif:maif /app

# Switch to non-root user
USER maif

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MAIF_ENVIRONMENT=production
ENV MAIF_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "from maif.health_check import HealthChecker; import sys; sys.exit(0 if HealthChecker.quick_check() else 1)"

# Default command
CMD ["python", "-m", "maif.cli", "serve"]

# Expose port for API
EXPOSE 8080