#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create directories
mkdir -p outputs logs

# Build the Docker image
echo -e "${YELLOW}Building enhanced people detection image...${NC}"
docker build -t people-detection-enhanced -f Dockerfile .

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Build successful!${NC}"

# Enable X11 for Docker
xhost +local:docker 2>/dev/null

# Determine which camera to use
if [ -e /dev/video0 ]; then
    CAMERA_DEVICE="/dev/video0"
    CAMERA_INDEX=0
elif [ -e /dev/video1 ]; then
    CAMERA_DEVICE="/dev/video1"
    CAMERA_INDEX=1
else
    echo -e "${RED}No camera detected at /dev/video0 or /dev/video1${NC}"
    exit 1
fi

echo -e "${GREEN}Using camera at $CAMERA_DEVICE (index $CAMERA_INDEX)${NC}"
echo -e "${YELLOW}Starting detection with logging and resource monitoring...${NC}"
echo -e "${YELLOW}Press ESC in the detection window to exit${NC}"

# Run the container
docker run --rm -it \
  --device=$CAMERA_DEVICE:$CAMERA_DEVICE \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/logs:/app/logs" \
  people-detection-enhanced --camera $CAMERA_INDEX

# Check if logs were created
LOG_COUNT=$(ls -1 logs/*.log 2>/dev/null | wc -l)
if [ $LOG_COUNT -gt 0 ]; then
    echo -e "${GREEN}Detection completed with logs saved to logs/ directory${NC}"
    echo -e "${YELLOW}Latest logs:${NC}"
    ls -lt logs/ | head -5
else
    echo -e "${YELLOW}Detection completed but no logs were found${NC}"
fi

# Check if output video was created
VIDEO_COUNT=$(ls -1 outputs/*.avi 2>/dev/null | wc -l)
if [ $VIDEO_COUNT -gt 0 ]; then
    echo -e "${GREEN}Output video saved to outputs/ directory${NC}"
    echo -e "${YELLOW}Latest videos:${NC}"
    ls -lt outputs/ | head -5
else
    echo -e "${YELLOW}No output videos were found${NC}"
fi
