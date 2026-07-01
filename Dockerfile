# Base Image
FROM python:3.11-slim

# Working Directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit Port
EXPOSE 8501

# Start Application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]