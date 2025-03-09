# Use an official lightweight Python image
FROM python:3.9.6

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Set environment variables for PyTorch MPS (Mac GPU)
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the script
CMD ["python", "run.py"]
