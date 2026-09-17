from TTS.api import TTS
import openwakeword
import whisper
import os
import json
from huggingface_hub import snapshot_download

def download_huggingface_model():
    """Download the base model from Hugging Face"""
    base_model = "unsloth/mistral-7b-v0.3-bnb-4bit"
    local_dir = "/app/models/text_to_command/base_model"
    
    print(f"Downloading the base model: {base_model} from Hugging Face...")
    
    try:
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=base_model,
            revision="main",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print("Model downloaded successfully.")
        
        # Update adapter_config.json
        update_adapter_config()
        return True
    except Exception as e:
        print(f"Failed to download the model: {e}")
        return False

def update_adapter_config():
    """Update base_model_name_or_path in adapter_config.json"""
    adapter_config_path = "/app/models/text_to_command/adapter_model/adapter_config.json"
    
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
            
            config["base_model_name_or_path"] = "/app/models/text_to_command/base_model"
            
            with open(adapter_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Updated base_model_name_or_path in {adapter_config_path}.")
        except Exception as e:
            print(f"Failed to update adapter config: {e}")

def download_models():
    """Download and cache models locally"""
    
    # Create models directory
    models_dir = "/app/models/speech_to_text"
    whisper_dir = os.path.join(models_dir, "whisper")
    os.makedirs(whisper_dir, exist_ok=True)
    
    print("Downloading TTS model...")
    tts_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
    print("TTS model downloaded and cached.")
    
    print("Downloading wake word model...")
    openwakeword.utils.download_models(["hey_jarvis_v0.1"])
    print("Wake word model downloaded and cached.")
    
    print("Downloading Whisper model...")
    whisper_model = whisper.load_model("base.en", download_root=whisper_dir)
    print("Whisper model downloaded and cached.")
    
    # Download Hugging Face model
    if not download_huggingface_model():
        print("Failed to download Hugging Face model, but continuing with other models.")
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_models()