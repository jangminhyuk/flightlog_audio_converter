import os
from pydub import AudioSegment
import torchaudio
from df import enhance, init_df  # Correct module for DeepFilterNet

# Initialize DeepFilterNet
model, df_state, _ = init_df()  # Load the default model
print("DeepFilterNet initialized successfully.")

# Define folder paths
INPUT_M4A_FOLDER = "real_flight_data_1214/audio_synced"
WAV_FOLDER = "real_flight_data_1214/wav_files"
DENOISED_FOLDER = "real_flight_data_1214/deepfilternet_denoised"

os.makedirs(WAV_FOLDER, exist_ok=True)
os.makedirs(DENOISED_FOLDER, exist_ok=True)

# Step 1: Convert .m4a to .wav
def convert_m4a_to_wav(input_file, output_file):
    try:
        audio = AudioSegment.from_file(input_file, format="m4a")
        audio.export(output_file, format="wav")
        print(f"Converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error converting {input_file}: {e}")

# Step 2: Apply DeepFilterNet
def denoise_with_deepfilternet(input_wav, output_wav):
    try:
        # Load the audio using torchaudio
        noisy_audio, sr = torchaudio.load(input_wav)

        # Enhance audio using DeepFilterNet
        enhanced_audio = enhance(model, df_state, noisy_audio)

        # Save the enhanced audio back to a .wav file
        torchaudio.save(output_wav, enhanced_audio, sample_rate=sr)
        print(f"DeepFilterNet denoised audio saved to {output_wav}")
    except Exception as e:
        print(f"Error processing {input_wav}: {e}")

# Step 3: Process all .m4a files
def process_audio_files(m4a_folder, wav_folder, denoised_folder):
    # Step 3.1: Convert all .m4a files to .wav
    for file_name in os.listdir(m4a_folder):
        if file_name.endswith(".m4a"):
            m4a_path = os.path.join(m4a_folder, file_name)
            wav_path = os.path.join(wav_folder, file_name.replace(".m4a", ".wav"))
            convert_m4a_to_wav(m4a_path, wav_path)

    # Step 3.2: Apply DeepFilterNet denoising
    for file_name in os.listdir(wav_folder):
        if file_name.endswith(".wav"):
            input_path = os.path.join(wav_folder, file_name)
            output_path = os.path.join(denoised_folder, file_name.replace(".wav", "_denoised.wav"))
            print(f"Processing {file_name}...")
            denoise_with_deepfilternet(input_path, output_path)

# Main process
if __name__ == "__main__":
    process_audio_files(INPUT_M4A_FOLDER, WAV_FOLDER, DENOISED_FOLDER)
