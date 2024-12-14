import os
import torchaudio
import torch
from pydub import AudioSegment
from speechbrain.inference import SpectralMaskEnhancement

# Define folders
INPUT_M4A_FOLDER = "real_flight_data_1214/audio_synced"  # Input .m4a files
WAV_FOLDER = "real_flight_data_1214/wav_files"           # Intermediate .wav files
OUTPUT_FOLDER = "real_flight_data_1214/speechbrain_denoised"  # Output denoised .wav files

# Ensure folders exist
os.makedirs(WAV_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the speech enhancement model
enhancer = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/speech_enhancement"
)

# Step 1: Convert .m4a to .wav
def convert_m4a_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")
    print(f"Converted {input_file} to {output_file}")

# Step 2: Apply SpeechBrain denoising
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
    enhanced_audio = enhancer.enhance_batch(noisy_audio, lengths=lengths)

    # Save the enhanced audio
    torchaudio.save(output_file, enhanced_audio.cpu(), sample_rate)
    print(f"Denoised audio saved to {output_file}")

# Step 3: Process all .m4a files
def process_audio_files(m4a_folder, wav_folder, output_folder):
    # Step 3.1: Convert all .m4a to .wav
    for file_name in os.listdir(m4a_folder):
        if file_name.endswith(".m4a"):
            m4a_path = os.path.join(m4a_folder, file_name)
            wav_path = os.path.join(wav_folder, file_name.replace(".m4a", ".wav"))
            try:
                convert_m4a_to_wav(m4a_path, wav_path)
            except Exception as e:
                print(f"Error converting {file_name}: {e}")

    # Step 3.2: Perform SpeechBrain denoising on the .wav files
    for file_name in os.listdir(wav_folder):
        if file_name.endswith(".wav"):
            input_path = os.path.join(wav_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".wav", "_denoised.wav"))
            print(f"Processing {file_name}...")
            try:
                denoise_with_speechbrain(input_path, output_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Main process
if __name__ == "__main__":
    process_audio_files(INPUT_M4A_FOLDER, WAV_FOLDER, OUTPUT_FOLDER)
