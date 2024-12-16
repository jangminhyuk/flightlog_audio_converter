import os
import librosa
import soundfile as sf
import torch
from asteroid.models import BaseModel
import numpy as np

# Define constants
INPUT_FOLDER = "testset_1216/testset_noisy"
OUTPUT_FOLDER = "testset_1216/asteroid_retrained_denoised"
BEST_MODEL_PATH = "dccrnet_best_model.pth"  # Path to your saved model
SAMPLE_RATE = 16000  # Fixed sample rate
SEGMENT_DURATION = 5  # Duration of each segment in seconds
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION  # Number of samples per segment

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the pretrained model (on CPU)
def load_model(model_path):
    """
    Load the pre-trained model for denoising.
    """
    model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode
    return model

# Process long audio files
def process_long_audio(model, input_file, output_file, sample_rate=SAMPLE_RATE, segment_samples=SEGMENT_SAMPLES):
    """
    Process a long audio file by slicing it into smaller segments, enhancing each segment,
    and combining the enhanced segments into a single output file.
    """
    # Load the long noisy audio
    noisy, _ = librosa.load(input_file, sr=sample_rate)
    
    # Slice the audio into smaller segments
    num_segments = int(np.ceil(len(noisy) / segment_samples))
    enhanced_audio = []

    print(f"Processing {input_file} into {num_segments} segments...")
    
    for i in range(num_segments):
        start = i * segment_samples
        end = min((i + 1) * segment_samples, len(noisy))
        segment = noisy[start:end]

        # Pad the last segment if it's shorter than segment_samples
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))

        # Convert to tensor
        segment_tensor = torch.tensor(segment).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, time]

        # Enhance the segment using the model
        with torch.no_grad():
            enhanced_tensor = model(segment_tensor)
        enhanced_segment = enhanced_tensor.squeeze().numpy()

        # Append to the full enhanced audio
        enhanced_audio.append(enhanced_segment[:end - start])  # Remove padding if any

    # Combine all enhanced segments
    enhanced_audio = np.concatenate(enhanced_audio)

    # Save the enhanced audio to output_file
    sf.write(output_file, enhanced_audio, sample_rate)
    print(f"Enhanced audio saved to {output_file}")

# Main script
if __name__ == "__main__":
    # Load the model
    model = load_model(BEST_MODEL_PATH)

    # Process each file in the input folder
    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.endswith(".wav"):
            input_file = os.path.join(INPUT_FOLDER, file_name)
            output_file = os.path.join(OUTPUT_FOLDER, file_name)
            
            # Process and save the enhanced audio
            process_long_audio(model, input_file, output_file)
