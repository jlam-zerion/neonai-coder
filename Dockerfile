# Use Ubuntu as base image
FROM --platform=linux/amd64 ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies and Python 3
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    curl \
    git \
    xz-utils \
    unzip \
    sudo \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python3
RUN ln -s /usr/bin/python3 /usr/bin/python

########### Install adtional packges here #########
# Install Node.js and npm (latest LTS) using package manager
RUN mkdir -p /etc/apt/keyrings \
    && wget -O- https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

# Install Next.js globally
RUN npm install -g next

# Install Flutter
# RUN git clone https://github.com/flutter/flutter.git -b stable /usr/opt/flutter
# ENV PATH="$PATH:/usr/opt/flutter/bin"
# RUN chown -R $USER:$USER /usr/opt/flutter
# RUN flutter precache
########### Install adtional packges here #########

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for file uploads and outputs
RUN mkdir -p input output

# Run the application
CMD ["python3", "main.py"]