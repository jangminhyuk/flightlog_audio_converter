import os
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import noisereduce as nr

# Define folder paths
INPUT_M4A_FOLDER = "real_flight_data_1214/audio_synced"
WAV_FOLDER = "real_flight_data_1214/wav_files"
DENOISED_FOLDER = "real_flight_data_1214/noisereduce_denoised"

os.makedirs(WAV_FOLDER, exist_ok=True)
os.makedirs(DENOISED_FOLDER, exist_ok=True)

# Step 1: Convert .m4a to .wav
def convert_m4a_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")
    print(f"Converted {input_file} to {output_file}")

# Step 2: Apply noise reduction
def reduce_noise(input_wav, output_wav):
    # Load the audio file
    data, rate = sf.read(input_wav)

    # Ensure mono audio if stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Ensure the data is a numpy array of type float32
    data = np.asarray(data, dtype=np.float32)

    print(f"Data shape: {data.shape}, Sample rate: {rate}")

    # Perform noise reduction
    try:
        reduced_noise = nr.reduce_noise(y=data, y_noise=None, sr=rate)
    except Exception as e:
        print(f"Error during noise reduction: {e}")
        return

    # Save the denoised audio
    sf.write(output_wav, reduced_noise, rate)
    print(f"Noise reduction completed for {input_wav}. Saved as {output_wav}")

# Step 3: Process all .m4a files
def process_audio_files(m4a_folder, wav_folder, denoised_folder):
    # Step 3.1: Convert all .m4a files to .wav
    for file_name in os.listdir(m4a_folder):
        if file_name.endswith(".m4a"):
            m4a_path = os.path.join(m4a_folder, file_name)
            wav_path = os.path.join(wav_folder, file_name.replace(".m4a", ".wav"))
            try:
                convert_m4a_to_wav(m4a_path, wav_path)
            except Exception as e:
                print(f"Error converting {file_name}: {e}")

    # Step 3.2: Apply noise reduction on .wav files
    for file_name in os.listdir(wav_folder):
        if file_name.endswith(".wav"):
            input_path = os.path.join(wav_folder, file_name)
            output_path = os.path.join(denoised_folder, file_name.replace(".wav", "_denoised.wav"))
            print(f"Processing {file_name}...")
            try:
                reduce_noise(input_path, output_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Main process
if __name__ == "__main__":
    process_audio_files(INPUT_M4A_FOLDER, WAV_FOLDER, DENOISED_FOLDER)
