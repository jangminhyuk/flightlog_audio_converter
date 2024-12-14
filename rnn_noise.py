import os
import wave
import rnnoise

#CODE NOT WORKING

# Step 1: Define input and output directories
INPUT_WAV_FOLDER = "real_flight_data_1214/wav_files"
DENOISED_FOLDER = "real_flight_data_1214/rnnoise_denoised"

os.makedirs(DENOISED_FOLDER, exist_ok=True)

# Step 2: Apply RNNoise
def denoise_with_rnnoise(input_wav, output_wav):
    try:
        denoiser = rnnoise.RNNoise()
        with wave.open(input_wav, "rb") as wav_in:
            frames = wav_in.readframes(wav_in.getnframes())
            params = wav_in.getparams()

        # Apply noise suppression
        denoised_frames = denoiser.filter(frames)

        # Save the enhanced audio
        with wave.open(output_wav, "wb") as wav_out:
            wav_out.setparams(params)
            wav_out.writeframes(denoised_frames)

        print(f"RNNoise denoised audio saved to {output_wav}")
    except Exception as e:
        print(f"Error processing {input_wav}: {e}")

# Step 3: Process all .wav files
def process_audio_files_rnnoise(wav_folder, denoised_folder):
    for file_name in os.listdir(wav_folder):
        if file_name.endswith(".wav"):
            input_path = os.path.join(wav_folder, file_name)
            output_path = os.path.join(denoised_folder, file_name.replace(".wav", "_denoised.wav"))
            print(f"Processing {file_name} with RNNoise...")
            denoise_with_rnnoise(input_path, output_path)

# Run the RNNoise processing
process_audio_files_rnnoise(INPUT_WAV_FOLDER, DENOISED_FOLDER)
