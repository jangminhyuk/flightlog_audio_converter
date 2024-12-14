import os
from pydub import AudioSegment
import noisereduce as nr
import soundfile as sf
import numpy as np

## NOISEREDUCE library - Spectral gating-based denoising model

# Step 1: Define folder paths
INPUT_FOLDER = "real_flight_data/audio_synced"  # Folder containing .m4a files
WAV_FOLDER = "real_flight_data/wav_files"  # Folder to save .wav files
DENOISED_FOLDER = "real_flight_data/denoised_files"  # Folder to save denoised .wav files

# Ensure folders exist
os.makedirs(WAV_FOLDER, exist_ok=True)
os.makedirs(DENOISED_FOLDER, exist_ok=True)

# Step 2: Convert .m4a to .wav
def convert_m4a_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")
    print(f"Converted {input_file} to {output_file}")

# Step 3: Perform noise reduction for the entire duration
def reduce_noise(input_wav, output_wav):
    # Load the audio file
    data, rate = sf.read(input_wav)
    
    # Use the first 1 second as noise sample
    noise_duration_seconds = 1  # Modify this if needed
    noise_sample = data[:int(noise_duration_seconds * rate)]
    
    # Denoise the entire audio
    reduced_noise = nr.reduce_noise(y=data, sr=rate, y_noise=noise_sample)
    
    # Save the denoised audio
    sf.write(output_wav, reduced_noise, rate)
    print(f"Noise reduction completed for {input_wav}. Saved as {output_wav}")

# Step 4: Process all .m4a files
def process_audio_files(input_folder, wav_folder, denoised_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".m4a"):
            input_file_path = os.path.join(input_folder, file_name)
            
            # Convert to .wav and save in WAV_FOLDER
            wav_file_path = os.path.join(wav_folder, file_name.replace(".m4a", ".wav"))
            convert_m4a_to_wav(input_file_path, wav_file_path)
            
            # Perform noise reduction and save in DENOISED_FOLDER
            denoised_file_path = os.path.join(denoised_folder, file_name.replace(".m4a", "_denoised.wav"))
            reduce_noise(wav_file_path, denoised_file_path)

# Main process
if __name__ == "__main__":
    process_audio_files(INPUT_FOLDER, WAV_FOLDER, DENOISED_FOLDER)
