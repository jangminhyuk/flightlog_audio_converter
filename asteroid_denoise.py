import os
import torch
import torchaudio
from asteroid.models import BaseModel

# Use the JorisCos DCCRNet-based model for single-speaker enhancement at 16kHz
model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")

INPUT_FOLDER = "real_flight_data_1214/wav_files"
OUTPUT_FOLDER = "real_flight_data_1214/asteroid_denoised"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def denoise_with_asteroid(input_file, output_file):
    # Load the audio file
    waveform, sr = torchaudio.load(input_file)
    
    # If stereo, convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # [1, time]

    # Resample if needed to 16 kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000

    # Model expects [batch, time]. We have [1, time], which is fine.
    with torch.no_grad():
        # separate() returns [batch, n_src, time]. Here n_src=1, single source.
        enhanced = model.separate(waveform)
        enhanced = enhanced[:, 0, :]  # shape: [1, time]

    # Save the enhanced audio
    torchaudio.save(output_file, enhanced.cpu(), sr)
    print(f"Denoised audio saved to {output_file}")

def process_audio_files(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".wav", "_denoised.wav"))
            print(f"Processing {file_name}...")
            try:
                denoise_with_asteroid(input_path, output_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    process_audio_files(INPUT_FOLDER, OUTPUT_FOLDER)
