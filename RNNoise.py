import soundfile as sf
from rnnoise import RNNoise

def denoise_with_rnnoise(input_file, output_file):
    # Load audio
    data, samplerate = sf.read(input_file)
    
    # Instantiate RNNoise
    denoiser = RNNoise()
    
    # Process audio frame-by-frame (RNNoise works best on short frames)
    frame_size = 480  # 30ms at 16kHz
    out_data = []
    for i in range(0, len(data), frame_size):
        frame = data[i:i+frame_size]
        if len(frame) < frame_size:
            # Pad the last frame if needed
            frame = list(frame) + [0]*(frame_size - len(frame))
        denoised_frame = denoiser.filter(frame)
        out_data.extend(denoised_frame)
    
    # Save denoised audio
    sf.write(output_file, out_data, samplerate)
    print(f"Denoised audio saved to {output_file}")
