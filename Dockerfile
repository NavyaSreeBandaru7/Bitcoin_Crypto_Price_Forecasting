# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data results logs cache && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('✅ Container is healthy')" || exit 1

# Default command
CMD ["python", "main.py"]

# -------------------------------------------------------------------
# GPU-enabled variant
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

# Install Python 3.9
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.9 \
        python3.9-pip \
        python3.9-dev \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CUDA_VISIBLE_DEVICES=0

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN python -m pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Copy and install requirements
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories and set permissions
RUN mkdir -p models data results logs cache && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Verify CUDA installation
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Expose port
EXPOSE 8000

# Health check with GPU verification
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); print('✅ GPU container healthy')" || exit 1

# Default command
CMD ["python", "main.py"]

# -------------------------------------------------------------------
# Development variant
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install Jupyter extensions
RUN pip install --no-cache-dir \
    jupyterlab \
    ipywidgets \
    jupyter-dash

# Expose additional ports for Jupyter
EXPOSE 8888 8050

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# -------------------------------------------------------------------
# Production API variant
FROM base as api

# Install additional API dependencies
RUN pip install --no-cache-dir \
    fastapi==0.85.0 \
    uvicorn[standard]==0.18.0 \
    redis==4.3.0 \
    celery==5.2.0 \
    gunicorn==20.1.0

# Copy API-specific files
COPY api/ ./api/

# Expose API port
EXPOSE 8000

# Production API command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "api.main:app"]
