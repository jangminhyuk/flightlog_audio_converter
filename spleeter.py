## Spleeter model
from spleeter.separator import Separator

# Step 1: Initialize Spleeter
separator = Separator('spleeter:2stems')  # Separate vocals and accompaniment

# Step 2: Denoise audio
def denoise_with_spleeter(input_file, output_file):
    separator.separate_to_file(input_file, output_file)
    print(f"Separated and denoised audio saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_file = "real_flight_data/audio_synced/noisy_audio.m4a"
    output_folder = "real_flight_data/spleeter_denoised/"
    denoise_with_spleeter(input_file, output_folder)
