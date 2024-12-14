import os
import torchaudio
import torch
from speechbrain.inference import SpectralMaskEnhancement

# Define folders
INPUT_FOLDER = "real_flight_data/wav_files"
OUTPUT_FOLDER = "real_flight_data/speechbrain_denoised"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the speech enhancement model
enhancer = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/speech_enhancement"
)

def denoise_with_speechbrain(input_file, output_file):
    # Load the audio
    noisy_audio, sample_rate = torchaudio.load(input_file)
    
    # Ensure mono audio: if stereo, average the channels
    if noisy_audio.shape[0] > 1:
        noisy_audio = torch.mean(noisy_audio, dim=0, keepdim=True)  # [1, time]

    # Resample to 16 kHz if necessary
    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        noisy_audio = resample_transform(noisy_audio)
        sample_rate = 16000

    # Set lengths as a relative fraction of the full audio. Full length = 1.0
    lengths = torch.tensor([1.0], dtype=torch.float32)

    # Apply enhancement
    # No need to convert lengths from samples or seconds. Just indicate full length with 1.0.
    enhanced_audio = enhancer.enhance_batch(noisy_audio, lengths=lengths)

    # Save the enhanced audio
    torchaudio.save(output_file, enhanced_audio.cpu(), sample_rate)
    print(f"Denoised audio saved to {output_file}")

def process_audio_files(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".wav", "_denoised.wav"))
            print(f"Processing {file_name}...")
            try:
                denoise_with_speechbrain(input_path, output_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    process_audio_files(INPUT_FOLDER, OUTPUT_FOLDER)
