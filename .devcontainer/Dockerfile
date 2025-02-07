# Use an official Python runtime as a parent image
FROM python:3.11

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install any OS-level dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy the requirements file and install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose any ports (if applicable) - not needed for this demo
