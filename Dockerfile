FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Rename files to standard names (handles both _V10_COMPLETE and _WORKING versions)
RUN if [ -f "divergence_scanner_V10_COMPLETE.py" ]; then \
        mv divergence_scanner_V10_COMPLETE.py divergence_scanner.py; \
    elif [ -f "divergence_scanner_WORKING.py" ]; then \
        mv divergence_scanner_WORKING.py divergence_scanner.py; \
    fi

RUN if [ -f "main_V10_COMPLETE.py" ]; then \
        mv main_V10_COMPLETE.py main.py; \
    elif [ -f "main_WORKING.py" ]; then \
        mv main_WORKING.py main.py; \
    fi

# Create directory for data
RUN mkdir -p /app/data

# Expose port (will be set by Railway/Render)
EXPOSE 8080

# Run the bot
CMD ["python", "main.py"]
