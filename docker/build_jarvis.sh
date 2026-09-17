#!/bin/bash

# Check if the tag parameter is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <tag>"
    exit 1
fi

TAG=$1

# Define the directories
BEST_MODEL_DIR="./research/text_to_command/fine-tuning/latest/best_model/"
ADAPTER_MODEL_DIR="./docker/models/text_to_command/adapter_model/"
VOICE_RESPONSES_SOURCE="./research/text_to_speech/voice_responses/"
VOICE_RESPONSES_DEST="./docker/models/text_to_speech/voice_responses/"

# Copy the best_model directory to adapter_model
if [ -d "$BEST_MODEL_DIR" ]; then
    mkdir -p "$ADAPTER_MODEL_DIR"
    cp -r "$BEST_MODEL_DIR"* "$ADAPTER_MODEL_DIR"
    echo "Copied $BEST_MODEL_DIR to $ADAPTER_MODEL_DIR."
else
    echo "Directory $BEST_MODEL_DIR does not exist."
    exit 1
fi

# Create voice_responses directory and copy content
if [ -d "$VOICE_RESPONSES_SOURCE" ]; then
    mkdir -p "$VOICE_RESPONSES_DEST"
    cp -r "$VOICE_RESPONSES_SOURCE"* "$VOICE_RESPONSES_DEST"
    echo "Copied voice responses from $VOICE_RESPONSES_SOURCE to $VOICE_RESPONSES_DEST."
else
    echo "Directory $VOICE_RESPONSES_SOURCE does not exist."
    exit 1
fi

# Build the Docker container with the full registry tag
echo "Building the Jarvis Docker container with tag: gitlab.lrz.de:5005/iac/jarvis:$TAG..."
cd docker
docker build -t gitlab.lrz.de:5005/iac/jarvis:$TAG .

if [ $? -ne 0 ]; then
    echo "Docker build failed."
    exit 1
fi

echo "Docker container built successfully with tag: gitlab.lrz.de:5005/iac/jarvis:$TAG."
