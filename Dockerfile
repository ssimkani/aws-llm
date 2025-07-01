# Python
FROM python:3.13.1-slim-bookworm

# Set working directory
WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# Copy requirements and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all other files into the container
COPY . .

# Set environment variable
ENV PYTHONPATH=/app

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "src/login.py", "--server.port=8501", "--server.address=0.0.0.0"]
