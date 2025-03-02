FROM python:3.9-slim

# Install full GUI support and monitoring tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgtk2.0-dev \
    pkg-config \
    python3-tk \
    python3-pil.imagetk \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    opencv-python==4.7.0.72 \
    numpy==1.23.5 \
    psutil==5.9.5

# Set up app directory structure
WORKDIR /app
RUN mkdir -p /app/outputs /app/logs

# Copy detector script
COPY enhanced_detector.py /app/

# Set the entry point
ENTRYPOINT ["python", "enhanced_detector.py"]

# Default to camera 0
CMD ["--camera", "0"]
