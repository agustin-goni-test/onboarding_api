FROM python:3.11-slim

WORKDIR /app

# Prevent Python from buffering logs (important for Cloudwatch)
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (informational)
EXPOSE 8000

# Start FASTAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]