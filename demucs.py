import os
import subprocess

# Denoising using demucs

# Step 1: Define input and output folders
INPUT_FOLDER = "testset_1216/testset_noisy"
OUTPUT_FOLDER = "testset_1216/demucs_denoised"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Step 2: Run Demucs on each file
def denoise_with_demucs(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        #if file_name.endswith(".m4a"):
        if file_name.endswith(".wav"):
            input_file = os.path.join(input_folder, file_name)
            #output_path = os.path.join(output_folder, file_name.replace(".m4a", "_denoised.wav"))
            output_path = os.path.join(output_folder, file_name.replace(".wav", "_denoised.wav"))
            
            # Run Demucs using subprocess
            subprocess.run([
                "demucs", "-n", "htdemucs", input_file, "--two-stems=vocals",
                "-o", output_folder
            ])
            print(f"Denoised {file_name} using Demucs. Output saved to {output_path}")

# Main process
if __name__ == "__main__":
    denoise_with_demucs(INPUT_FOLDER, OUTPUT_FOLDER)
