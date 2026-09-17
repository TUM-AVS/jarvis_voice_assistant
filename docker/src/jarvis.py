import openwakeword
from openwakeword.model import Model as WakeWordModel  # Rename to avoid conflict
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import rclpy
from rclpy.node import Node
from TTS.api import TTS
import json
import pyaudio
import whisper
import numpy as np
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

from std_msgs.msg import String

import sys
sys.path.insert(1, '../text_to_command/')
from common import command_to_index

import sounddevice as sd
import soundfile as sf

from text_to_num import alpha2digit
import re

# Audio settings
CHUNK = 1024  # Smaller chunk size
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Wake word model
WAKE_WORD_MODEL_NAME = "hey_jarvis_v0.1"


class VoiceAudioPublisher(Node):
    def __init__(self, ttc_model, ttc_tokenizer, tts_model, whisper_model, gui=None):
        super().__init__('voice_audio_publisher')
        self.gui = gui
        self.command_pub = self.create_publisher(String, 'voice_command', 10)
        
        # Store models as instance variables
        self.ttc_model = ttc_model
        self.ttc_tokenizer = ttc_tokenizer
        self.tts_model = tts_model
        self.whisper_model = whisper_model

        # Audio queue for Whisper
        self.audio_queue = queue.Queue()
        self.transcription_result = None
        self.transcription_ready = threading.Event()

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self._stop_event = threading.Event()  # added

    def open_audio_stream(self):
        # Open audio stream
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)
        
    def close_audio_stream(self):
        # Close audio stream
        self.stream.stop_stream()
        self.stream.close()

    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice audio stream"""
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def whisper_transcription_thread(self, duration=5):
        """Thread function to handle Whisper transcription"""
        buffer = np.zeros(0, dtype=np.float32)
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                data = self.audio_queue.get(timeout=0.1)
                buffer = np.concatenate([buffer, data.flatten()])
            except queue.Empty:
                continue

        if len(buffer) > 0:
            # Normalize audio to [-1, 1] range
            if buffer.dtype == np.int16:
                buffer = buffer.astype(np.float32) / 32768.0
            
            # Run Whisper transcription
            result = self.whisper_model.transcribe(buffer, language="en", fp16=False)
            self.transcription_result = result["text"].strip()
        else:
            self.transcription_result = ""
        
        self.transcription_ready.set()

    def listen_with_whisper(self, duration=5):
        """Listen for audio and transcribe with Whisper"""
        # Clear previous results
        self.transcription_result = None
        self.transcription_ready.clear()
        
        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Start audio stream with sounddevice
        stream = sd.InputStream(
            samplerate=RATE, 
            channels=1,
            callback=self.audio_callback, 
            blocksize=CHUNK,
            dtype=np.float32
        )

        # Start transcription thread
        transcription_thread = threading.Thread(
            target=self.whisper_transcription_thread, 
            args=(duration,)
        )

        with stream:
            transcription_thread.start()
            transcription_thread.join()

        return self.transcription_result

    def text_to_command(self, text):
        """Convert text to command using the model"""
        command_list = list(command_to_index.keys())
        command_list.remove("CRANK_REQUEST") # not supported
        
        formatted_prompt = f"<s>[INST] {text} [/INST]"
        inputs = self.ttc_tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        generated = self.ttc_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,  # make deterministic to avoid sampling variance
        )

        new_tokens = generated[0, input_ids.shape[-1]:]
        generated_text = self.ttc_tokenizer.decode(new_tokens, skip_special_tokens=True)

        try:
            if generated_text in command_list:
                return generated_text
            else:
                raise Exception("Command not recognized")
        except Exception as e:
            return "ERROR"
        
    def play_audio(self, wav_data):
        """Play audio using sounddevice with better error handling"""
        try:
            if wav_data is not None and len(wav_data) > 0:
                # Play the audio with explicit blocksize to prevent underruns
                sd.play(wav_data, samplerate=22050, blocksize=1024)
                sd.wait()  # Wait for playback to complete
        except Exception as e:
            print(f"Error playing audio: {e}")

    def run_voice_loop(self):
        print("Listening... (Press Ctrl+C to stop)")
        if self.gui:
            self.gui.update_state("idle", "Say 'Hey Jarvis'")

        try:
            wake_word_detected = False
            open_wake_word_stream = True
            while not self._stop_event.is_set():  # changed
                if wake_word_detected:
                    print("Listening for command...")
                    if self.gui:
                        self.gui.update_state("listening", "Listening for command...")

                    # Use Whisper for speech recognition
                    recognized_text = self.listen_with_whisper(duration=5)

                    if recognized_text:
                        print(f"STT: {recognized_text}")
                        if self.gui:
                            self.gui.update_state("listening", f"Heard: \"{recognized_text}\"")

                        wake_word_detected = False
                        open_wake_word_stream = True  # Reset to wait for wake word again

                        # Process the recognized text to command
                        start = time.time()
                        command = self.text_to_command(recognized_text)
                        end = time.time()

                        # Extract numerical value if present
                        converted = alpha2digit(recognized_text, "en")
                        pattern = r"-?\d+(?:\.\d+)?"
                        numbers = re.findall(pattern, converted)
                        numbers = [int(n) if n.isdigit() else float(n) for n in numbers]
                        value = numbers[0] if numbers else ""
                        msg_command = command
                        msg_value = value

                        # Detect units
                        kmh_list = ["kilometers per hour", "km per hour", "km/h", "kph", "kmph", "kmh"]
                        mps_list = ["meters per second", "mps", "m/s"]
                        kmh = any(unit in recognized_text.lower() for unit in kmh_list)
                        mps = any(unit in recognized_text.lower() for unit in mps_list) and not kmh

                        if value != "" and command in ["SPEED_REQUEST", "GG_SCALE"]:
                            if command == "SPEED_REQUEST" and kmh:
                                msg_value = round(value / 3.6, 5)
                            msg_command = command + "\n" + str(msg_value) + "\nCHANGE"

                        if (command in ["SPEED_REQUEST", "GG_SCALE"] and value == "") or (
                            command == "SPEED_REQUEST" and not (kmh or mps)
                        ):
                            print("No numerical value detected for the command.")
                            if self.gui:
                                self.gui.update_state("error", "No numerical value detected.")
                            wav = sf.read(f"/app/models/text_to_speech/voice_responses/ERROR.wav", dtype="float32")[0]
                            self.play_audio(wav)
                            continue

                        print(f"TTC processing time: {end - start:.2f} seconds")
                        wav = None

                        if command in command_to_index.keys():
                            print(f"Command recognized: {command}")
                            if self.gui:
                                text_msg = f"Command [{command}] recognized"
                                if value != "":
                                    text_msg += f", value: {value}"
                                    if kmh:
                                        text_msg += " km/h"
                                    elif mps:
                                        text_msg += " m/s"
                                self.gui.update_state("success", text_msg)

                            wav = sf.read(f"/app/models/text_to_speech/voice_responses/{command}.wav", dtype="float32")[0]
                            command_wav = wav

                            if command == "SPEED_REQUEST":
                                value_wav = self.tts_model.tts("to " + str(value), speaker="p230")
                                max_val = np.max(np.abs(value_wav))
                                value_wav = value_wav / max_val * 0.95 if max_val > 0 else value_wav
                                value_wav = np.array(value_wav, dtype=np.float32)

                                if kmh:
                                    unit_wav = sf.read(f"/app/models/text_to_speech/voice_responses/kmh.wav", dtype="float32")[0]
                                elif mps:
                                    unit_wav = sf.read(f"/app/models/text_to_speech/voice_responses/mps.wav", dtype="float32")[0]
                                wav = np.concatenate((command_wav, value_wav, unit_wav))

                            elif command == "GG_SCALE":
                                value_wav = self.tts_model.tts("to " + str(value), speaker="p230")
                                max_val = np.max(np.abs(value_wav))
                                value_wav = value_wav / max_val * 0.95 if max_val > 0 else value_wav
                                value_wav = np.array(value_wav, dtype=np.float32)
                                wav = np.concatenate((command_wav, value_wav))

                            wav = np.array(wav, dtype=np.float32)
                            self.play_audio(wav)

                            print("Please say 'yes' or 'no'")
                            if self.gui:
                                # Keep the recognized command text visible while waiting for confirmation
                                confirm_msg = text_msg + "\n\nPlease say 'yes' or 'no'"
                                self.gui.update_state("listening", confirm_msg)

                            confirmation_text = self.listen_with_whisper(duration=3)

                            if confirmation_text:
                                print(f"Confirmation STT: {confirmation_text}")
                                confirmation = confirmation_text.lower()
                                positive_responses = [
                                    "yes", "yeah", "yep", "confirm", "execute", "do it", "go", "okay", "ok", "sure"
                                ]
                                negative_responses = ["no", "nope", "cancel", "abort", "stop", "don't", "never"]

                                if any(resp in confirmation for resp in positive_responses):
                                    if self.gui:
                                        self.gui.update_state("success", "Command executed successfully.")
                                    print(f"Command confirmed: {msg_command}")
                                    msg = String()
                                    msg.data = msg_command
                                    self.command_pub.publish(msg)
                                    self.get_logger().info(f"Published command: {msg_command}")
                                    wav = sf.read(f"/app/models/text_to_speech/voice_responses/EXECUTED.wav", dtype="float32")[0]
                                    self.play_audio(wav)
                                elif any(resp in confirmation for resp in negative_responses):
                                    if self.gui:
                                        self.gui.update_state("idle", "Command cancelled.")
                                    print("Command cancelled.")
                                    wav = sf.read(f"/app/models/text_to_speech/voice_responses/CANCELLED.wav", dtype="float32")[0]
                                    self.play_audio(wav)
                                else:
                                    print(f"Unclear confirmation: {confirmation}")
                                    wav = sf.read(f"/app/models/text_to_speech/voice_responses/ERROR.wav", dtype="float32")[0]
                                    self.play_audio(wav)
                                    if self.gui:
                                        self.gui.update_state("error", "Unclear confirmation.")
                            else:
                                if self.gui:
                                    self.gui.update_state("error", "No confirmation detected.")
                        else:
                            print(f"Command not recognized: {command}")
                            if self.gui:
                                self.gui.update_state("error", "Error: Command not recognized.")
                            wav = sf.read(f"/app/models/text_to_speech/voice_responses/ERROR.wav", dtype="float32")[0]
                            self.play_audio(wav)

                    else:
                        print("No voice detected.")
                        if self.gui:
                            self.gui.update_state("error", "No voice detected.")
                else:
                    if open_wake_word_stream:
                        self.open_audio_stream()
                        open_wake_word_stream = False
                        self.wake_model = WakeWordModel(wakeword_models=[WAKE_WORD_MODEL_NAME])
                        print("Say 'Hey Jarvis'")
                        if self.gui:
                            self.gui.update_state("idle", "Say 'Hey Jarvis'")

                    # Read audio chunk
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    audio_array = np.frombuffer(data, dtype=np.int16)

                    # Get wake word predictions
                    prediction = self.wake_model.predict(audio_array)

                    # Check if "Jarvis" was detected
                    if prediction.get("hey_jarvis_v0.1", 0) > 0.5:
                        print("Jarvis wake word detected!")
                        wake_word_detected = True
                        self.close_audio_stream()
                        if self.gui:
                            self.gui.update_state("listening", "Hi Jarvis! Listening...")
                        wav = sf.read(f"/app/models/text_to_speech/voice_responses/HI.wav", dtype="float32")[0]
                        self.play_audio(wav)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()
            if self.gui:
                self.gui.update_state("idle", "Stopped.")

    def stop(self):
        """Signal loop to stop."""
        self._stop_event.set()

    def cleanup(self):
        """Release audio resources safely."""
        try:
            if hasattr(self, "stream") and self.stream is not None:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
        except Exception:
            pass
        try:
            if hasattr(self, "p") and self.p is not None:
                self.p.terminate()
        except Exception:
            pass



def load_whisper_model(model_name="base.en", cache_dir="/app/models/speech_to_text/whisper"):
    """Load Whisper model from cache or download if not present"""
    model_path = os.path.join(cache_dir, f"{model_name}.pt")
    
    if os.path.exists(model_path):
        print(f"Loading cached Whisper model from {model_path}")
        return whisper.load_model(model_path)
    else:
        print(f"Cached model not found, downloading Whisper model '{model_name}'")
        os.makedirs(cache_dir, exist_ok=True)
        return whisper.load_model(model_name, download_root=cache_dir)


class VoiceAssistantGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Jarvis Voice Assistant")
        self.root.geometry("400x450")
        self.root.resizable(False, False)
        self.root.configure(bg="#101010")

        # Load images (static states)
        self.images = {
            "idle": ImageTk.PhotoImage(Image.open("./images/logo_small.png").resize((200, 200), Image.LANCZOS)),
            "success": ImageTk.PhotoImage(Image.open("./images/logo_green.png").resize((200, 200), Image.LANCZOS)),
            "error": ImageTk.PhotoImage(Image.open("./images/logo_red.png").resize((200, 200), Image.LANCZOS)),
        }

        # Base image for pulsing in 'listening' state (use a higher-res source)
        self.listening_base = Image.open("./images/logo_big.png")  # PIL Image
        self.listening_photo = None  # will hold current PhotoImage frame

        # Pulse animation state
        self.pulsing = False
        self.pulse_job = None
        self.pulse_min = 160
        self.pulse_max = 240
        self.pulse_step = 8
        self.pulse_size = self.pulse_min
        self.pulse_dir = 1  # 1 = growing, -1 = shrinking

        # Fixed-size container so the status label doesn't shift when pulsing
        self.logo_frame = tk.Frame(self.root, width=self.pulse_max, height=self.pulse_max, bg="#101010")
        self.logo_frame.pack(pady=(30, 10))
        self.logo_frame.pack_propagate(False)  # keep frame size fixed

        # Logo label centered inside the fixed container
        self.logo_label = tk.Label(self.logo_frame, image=self.images.get("idle"), bg="#101010")
        self.logo_label.image = self.images.get("idle")  # keep reference
        self.logo_label.place(relx=0.5, rely=0.5, anchor="center")

        # Status label
        self.status_label = tk.Label(
            self.root, text="Say 'Hey Jarvis'",
            font=("Segoe UI", 14), fg="white", bg="#101010", wraplength=350, justify="center"
        )
        self.status_label.pack(pady=20)

        # Run on main thread now (do not start mainloop in a thread)
        # threading.Thread(target=self.root.mainloop, daemon=True).start()  # removed

    def run(self):
        self.root.mainloop()

    def update_state(self, state: str, text: str = ""):
        """Thread-safe UI update; schedules actual changes on the Tk thread."""
        self.root.after(0, lambda: self._update_state_on_ui(state, text))

    def _update_state_on_ui(self, state: str, text: str = ""):
        """Update GUI image and text on the Tk thread."""
        if state == "listening":
            # Start pulsing animation
            if not self.pulsing:
                self._start_pulsing()
        else:
            # Stop pulsing and set static image for the state
            self._stop_pulsing()
            if state in self.images:
                self.logo_label.configure(image=self.images[state])
                self.logo_label.image = self.images[state]

        if text:
            self.status_label.configure(text=text)

    def _start_pulsing(self):
        self.pulsing = True
        self.pulse_size = self.pulse_min
        self.pulse_dir = 1
        self._pulse_step()

    def _stop_pulsing(self):
        if self.pulsing and self.pulse_job is not None:
            self.root.after_cancel(self.pulse_job)
        self.pulsing = False
        self.pulse_job = None

    def _pulse_step(self):
        if not self.pulsing:
            return

        # Compute next size
        self.pulse_size += self.pulse_step * self.pulse_dir
        if self.pulse_size >= self.pulse_max:
            self.pulse_size = self.pulse_max
            self.pulse_dir = -1
        elif self.pulse_size <= self.pulse_min:
            self.pulse_size = self.pulse_min
            self.pulse_dir = 1

        # Create resized frame and update label
        frame = self.listening_base.resize((self.pulse_size, self.pulse_size), Image.LANCZOS)
        self.listening_photo = ImageTk.PhotoImage(frame)
        self.logo_label.configure(image=self.listening_photo)
        self.logo_label.image = self.listening_photo  # keep reference

        # Schedule next frame
        self.pulse_job = self.root.after(80, self._pulse_step)


def main():
    # Load the text to command model and tokenizer
    print("Loading the text to command model.")
    ttc_model, ttc_tokenizer = FastLanguageModel.from_pretrained(
        model_name = "/app/models/text_to_command/adapter_model",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True)

    ttc_tokenizer = get_chat_template(
        ttc_tokenizer,
        chat_template="mistral",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )

    FastLanguageModel.for_inference(ttc_model)
    print("Text to command model loaded successfully.")

    # Initialize TTS model
    print("Initializing TTS model.")
    tts_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
    print("TTS model initialized.")

    # Download the wake word model if not already present (should be cached)
    print("Downloading wake word model.")
    openwakeword.utils.download_models([WAKE_WORD_MODEL_NAME])
    print("Wake word model downloaded successfully.")

    # Load the Whisper model (STT)
    print("Loading Whisper model.")
    whisper_model = load_whisper_model("base.en", "/app/models/speech_to_text/whisper")
    print("Whisper model loaded successfully.")

    rclpy.init()
    try:
        gui = VoiceAssistantGUI()  # start GUI (main thread)
        node = VoiceAudioPublisher(ttc_model, ttc_tokenizer, tts_model, whisper_model, gui=gui)
        print("VoiceAudioPublisher node initialized.")

        # Start the voice loop in a background thread
        worker = threading.Thread(target=node.run_voice_loop, daemon=True)
        worker.start()

        # Clean shutdown when window closes
        def on_close():
            node.stop()
            gui.root.destroy()
        gui.root.protocol("WM_DELETE_WINDOW", on_close)

        # Enter Tk mainloop on the main thread
        gui.run()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'node' in locals():
            node.stop()
            node.cleanup()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()