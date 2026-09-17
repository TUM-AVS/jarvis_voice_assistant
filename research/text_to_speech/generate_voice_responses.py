import soundfile as sf
import numpy as np
import os
from TTS.api import TTS

command = {
    "HI" : "Hi!",

    "STOP" : "Do you want to stop driving?",
    "START" : "Do you want to start driving?",
    "ENGINE_STOP" : "Do you want to stop the engine?",
    "ENGINE_START" : "Do you want to start the engine?",
    # "CRANK_REQUEST" : "Do you want to initiate the engine cranking process?",
    "STOP_IN_PITLANE_TOGGLE_ON" : "Do you want to enter the pit lane and stop there?",
    "STOP_IN_PITLANE_TOGGLE_OFF" : "Do you want to cancel the pit lane stop mode?",
    "SPEED_BY_RC_TOGGLE_ON" : "Do you want to enable race control to manage the vehicle’s speed?",
    "SPEED_BY_RC_TOGGLE_OFF" : "Do you want to disable race control’s speed management?",
    "FLAG_BY_RC_TOGGLE_ON" : "Do you want to activate race control flag signaling?",
    "FLAG_BY_RC_TOGGLE_OFF" : "Do you want to disable race control flag signaling?",
    "INTO_PITLANE_TOGGLE_ON" : "Do you want to direct the vehicle to enter the pit lane?",
    "INTO_PITLANE_TOGGLE_OFF" : "Do you want to cancel pit lane entry mode?",
    "JOYSTICK_TOGGLE_ON" : "Do you want to enable joystick control of the vehicle?",
    "JOYSTICK_TOGGLE_OFF" : "Do you want to disable joystick control?",
    "SPEED_REQUEST" : "Do you want to request a speed change",
    "GG_SCALE" : "Do you want to adjust the GG scale",
    "ERROR" : "Command not recognized.",

    "mps" : "meters per second?",
    "kmh" : "kilometers per hour?",

    "EXECUTED" : "Command executed successfully.",
    "CANCELLED" : "Command cancelled.",
}


def trim_silence(audio, threshold=0.01):
    """Trim silence from the end of audio"""
    # Find the last non-silent sample
    for i in range(len(audio) - 1, -1, -1):
        if abs(audio[i]) > threshold:
            return audio[:i + 1]
    return audio


# Create output directory if it doesn't exist
os.makedirs("voice_responses", exist_ok=True)


# Generate and save audio file properly
tts_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)


# Generate and play audio for each command
for cmd, text in command.items():
    print(f"Generating audio for command '{cmd}'...")
    
    # Generate audio
    wav = tts_model.tts(text, speaker="p230")
    
    # Convert to proper format and trim silence
    wav_array = np.array(wav, dtype=np.float32)
    wav_trimmed = trim_silence(wav_array)
    
    # Normalize to use full dynamic range (makes it as loud as possible)
    max_val = np.max(np.abs(wav_trimmed))
    if max_val > 0:
        wav_amplified = wav_trimmed / max_val * 0.95  # 0.95 to avoid potential clipping
    else:
        wav_amplified = wav_trimmed
    
    sf.write(f"voice_responses/{cmd}.wav", wav_amplified, 22050)

