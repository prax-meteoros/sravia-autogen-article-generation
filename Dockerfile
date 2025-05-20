# Use official Python runtime base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY app.py .
COPY article_template.txt .

# Expose the port Flask runs on
EXPOSE 5002

# Run the app
CMD ["python", "app.py"]
