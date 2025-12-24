FROM python:3.11-slim

# Run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Create and switch to app directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Northflank uses PORT environment variable
EXPOSE 8080

# Run the bot
CMD ["python", "main.py"]
