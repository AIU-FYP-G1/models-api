FROM python:3.9

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# EB uses port 8080
EXPOSE 8080

# Get port from environment variable or default to 8080
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}