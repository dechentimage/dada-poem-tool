# Use official Python slim image
FROM python:3.10-slim

# Install tesseract OCR
RUN apt-get update && \
    apt-get install -y tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Download spaCy language models for German and English
RUN python -m spacy download de_core_news_sm && \
    python -m spacy download en_core_web_sm

# Copy application source
COPY . /app
WORKDIR /app

# Expose port for render (environment variable PORT is used by Render)
ENV PORT=8080

# Start the app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
