# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (FFmpeg, etc.)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY app.py .
COPY pipeline.py .
COPY debug_step_by_step.py .

# Create necessary directories
RUN mkdir -p input output transcripts temp logs

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper model to avoid downloading on first run
# This speeds up the first video processing significantly
RUN python -c "import whisper; whisper.load_model('small')" && \
    echo "Whisper model downloaded successfully"

# Expose port 5000 for Flask app
EXPOSE 5000

# Set environment to production
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check (optional but good practice)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000')" || exit 1

# Run the Flask app
CMD ["python", "app.py"]