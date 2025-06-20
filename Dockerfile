FROM python:3.13.1-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies (including git)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# Copy requirement files
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app into the container
COPY . .

# Expose Streamlit port
EXPOSE 8501
EXPOSE 11

# Run the app
CMD ["streamlit", "run", "src/login.py", "--server.port=8501", "--server.address=0.0.0.0"]
