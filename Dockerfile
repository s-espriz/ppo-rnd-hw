# Dockerfile
FROM python:3.8-slim-buster

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /task2

COPY . .

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg

# Install Jupyter
RUN pip install --upgrade pip && \
    pip install notebook jupyterlab

# OPTIONAL: install torch/gym here or in the notebook
RUN pip install --upgrade pip && pip install -r requirements.txt


# Set working directory
WORKDIR /task2

# Expose Jupyter port
EXPOSE 8888

# Run Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]

