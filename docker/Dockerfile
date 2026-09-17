FROM ros:humble-ros-base

# --------------------------
# System dependencies
# --------------------------
RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3-tk \
        python3-pip \
        git \
        cmake \
        build-essential \
        python3-colcon-common-extensions \
        alsa-utils \
        pulseaudio \
        pulseaudio-utils \
        libasound2-dev \
        portaudio19-dev \
        libportaudio2 \
        libportaudiocpp0 \
        ffmpeg \
        sox \
        espeak \
        espeak-data \
        libespeak1 \
        libespeak-dev \
        espeak-ng \
        espeak-ng-data \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# Python setup
# --------------------------
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3 -m pip install --upgrade pip==25.1.1

# --------------------------
# Environment variables
# --------------------------
ENV ESPEAK_DATA_PATH=/usr/share/espeak-ng-data
ENV PULSE_SERVER=unix:/run/user/1000/pulse/native

WORKDIR /app

# --------------------------
# Python dependencies
# --------------------------
COPY src/requirements.txt .
RUN python3 -m pip install -r requirements.txt
# Patch TRL to add VERSION file
RUN python3 -c "import pathlib; pathlib.Path('/usr/local/lib/python3.10/dist-packages/trl/../VERSION').write_text('0.22.0')"

# --------------------------
# Copy models and code
# --------------------------
COPY ./models/text_to_command/adapter_model ./models/text_to_command/adapter_model
COPY ./models/text_to_speech/voice_responses ./models/text_to_speech/voice_responses
COPY ./src/ ./src/
COPY ./src/images ./images

# --------------------------
# Download and cache models
# --------------------------
RUN python3 /app/src/download_models.py

# --------------------------
# Audio configuration
# --------------------------
RUN groupadd -r audio && \
    useradd -r -g audio -G audio,pulse-access audiouser || true
RUN echo "pcm.!default pulse" > /etc/asound.conf && \
    echo "ctl.!default pulse" >> /etc/asound.conf

# --------------------------
# Run Jarvis directly (unbuffered)
# --------------------------
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && python3 -u /app/src/jarvis.py"]
