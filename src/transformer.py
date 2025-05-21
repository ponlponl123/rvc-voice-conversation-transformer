import argparse
import datetime
import torch
import os.path
import librosa
import soundfile as sf
import numpy as np

try:
    from rvc.lib.vc_infer_pipeline import VC
    from rvc.lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid, 
        SynthesizerTrnMs256NSFsid
    )

except ImportError as e:
    print(f"Failed to import RVC core components from 'rvc' package. Error: {e}")
    print("Please ensure the 'rvc' package is correctly installed and its structure matches the expected imports.")
    print("If you installed 'rvc' from PyPI, it might be a simplified wrapper or a specific fork.")
    print("You might need to refer to the RVC library's documentation for correct usage.")
    import sys
    sys.exit(1) # Exit if core RVC components can't be found

from utils.loader import loader 

# --- Global/Cached Models (Good for performance) ---
global_vc_pipeline = None
global_device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_audio_file(file_path, target_sr=16000):
    # Load audio. RVC's internal functions will handle resampling to 16kHz if needed
    # for feature extraction, and then to the model's sample rate.
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    audio = audio.astype(np.float32)
    return audio, sr

def _perform_vc(model_data, audio, sr, f0_extractor_name='rmvpe', target_f0_shift=0, 
                output_sr=48000, index_path=None, index_rate=0.75, protect_f0=0.5):
    """
    Performs the actual RVC voice conversion using the installed 'rvc' library.
    
    Args:
        model_data: The loaded RVC model (dictionary from .pth file).
        audio: Input audio waveform (numpy array, ideally at 16kHz for best results with RVC).
        sr: Sample rate of the input audio.
        f0_extractor_name: F0 extraction method (e.g., 'rmvpe', 'crepe').
        target_f0_shift: Pitch shift in semitones.
        output_sr: Desired output sample rate (e.g., 48000).
        index_path: Path to the FAISS index file for the model.
        index_rate: Weight of the index in timbre transfer (0 to 1).
        protect_f0: F0 protect parameter (0.0 to 0.5 for V2, higher values protect more of original F0).
    """
    global global_vc_pipeline, global_device

    print(f"  Performing voice conversion using 'rvc' package on {global_device}...")
    print(f"  Target F0 shift: {target_f0_shift} semitones")
    print(f"  Output Sample Rate: {output_sr} Hz")
    print(f"  F0 extractor: {f0_extractor_name}")
    print(f"  Index path: {index_path if index_path else 'None'}, Index rate: {index_rate}")
    print(f"  F0 Protect: {protect_f0}")

    # --- Initialize VC pipeline if not already ---
    if global_vc_pipeline is None:
        print(f"  Initializing RVC VC pipeline from 'rvc' package...")
        # Determine the model's actual sampling rate for the VC class initialization
        # RVC models are trained at a specific sample rate (e.g., 40000, 48000).
        model_samplerate = model_data.get("sample_rate", 48000) # Get original training SR of the model
        
        # The `VC` class constructor might also need the 'is_half' argument for half-precision
        is_half = (global_device == "cuda")

        hubert_model_path = "assets/hubert/hubert_base.pt"
        rmvpe_model_path = "assets/rmvpe/rmvpe.pt" # Used by RMVPE F0 method
        crepe_model_path = None # Crepe often doesn't need a specific .pt file in assets
        vocoder_model_path = "assets/hifigan/hifigan_v2.pth" # HiFi-GAN for synthesis

        # Check if assets exist where expected
        if not os.path.exists(hubert_model_path):
            print(f"WARNING: HuBERT model not found at {hubert_model_path}. RVC may fail or use default.")
        if f0_extractor_name == 'rmvpe' and not os.path.exists(rmvpe_model_path):
            print(f"WARNING: RMVPE model not found at {rmvpe_model_path}. RMVPE F0 extraction may fail.")
        if not os.path.exists(vocoder_model_path):
            print(f"WARNING: Vocoder model not found at {vocoder_model_path}. RVC may fail to synthesize audio.")

        # Initialize the VC pipeline
        # The VC class typically takes the *path* to the RVC model file.
        # It handles loading the model architecture and weights itself.
        global_vc_pipeline = VC(
            model_path=model_data.path, # Use the original path from loader.py
            # Pass asset paths. The VC class uses these to load its sub-models.
            hubert_model_path=hubert_model_path,
            rmvpe_model_path=rmvpe_model_path,
            crepe_model_path=crepe_model_path, # Pass None if not used by Crepe setup
            vocoder_model_path=vocoder_model_path,
            device=global_device,
            is_half=is_half
        )
        print("  RVC VC pipeline initialized from 'rvc' package.")
    speaker_id = 0 
    
    # Convert input audio to 16kHz (RVC's internal processing SR) if not already.
    # The VC class should handle this, but explicit check for safety.
    if sr != 16000:
        print(f"  Resampling input audio from {sr} Hz to 16000 Hz for RVC processing...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000 # Update current sample rate

    # The `vc_single` method expects raw audio (numpy array)
    audio_opt = global_vc_pipeline.vc_single(
        audio=audio,
        f0_up_key=target_f0_shift,
        sid=speaker_id,
        f0_method=f0_extractor_name,
        index_file=index_path, # Pass the FAISS index path
        index_rate=index_rate,
        protect=protect_f0,
        resample_sr=output_sr, # VC class can handle final resampling to output_sr
        # Other parameters often available:
        # filter_radius=3,
        # rms_mix_rate=0.0, # Volume envelope mixing
    )

    # `audio_opt` should already be a numpy array at `output_sr` if `resample_sr` was set.
    converted_audio = audio_opt 
    
    return converted_audio

def main():
    parser = argparse.ArgumentParser(description='Perform RVC Voice Conversion.')
    parser.add_argument("-m", "--model", help="Load RVC model from file", required=True)
    parser.add_argument("-f", "--file", help="Input audio file to process", required=True)
    parser.add_argument("-o", "--output", help="Output audio file path (e.g., output.wav)")
    parser.add_argument("-k", "--key", type=int, default=0, 
                        help="Pitch shift in semitones (e.g., +12 for octave up, -12 for octave down).")
    parser.add_argument("-e", "--f0_extractor", type=str, default="rmvpe",
                        help="F0 extraction method ('rmvpe' or 'crepe').")
    parser.add_argument("-osr", "--output_sample_rate", type=int, default=48000,
                        help="Output sample rate for the converted audio.")
    parser.add_argument("-idx", "--index_path", type=str, default=None,
                        help="Path to the FAISS index file (e.g., 'logs/my_model/added_IVF1972_Flat_nprobe_1_my_model_v2.index').")
    parser.add_argument("-idx_rate", "--index_rate", type=float, default=0.75,
                        help="Weight of the index in timbre transfer (0.0 to 1.0). Higher uses index more.")
    parser.add_argument("-p", "--protect_f0", type=float, default=0.5,
                        help="F0 protect parameter (0.0 to 0.5 for V2, higher values protect original F0 more).")


    args = parser.parse_args()

    convert(
        model_path=args.model, 
        input_file=args.file, 
        output_file=args.output, 
        pitch_shift_key=args.key,
        f0_extractor_name=args.f0_extractor,
        output_sr=args.output_sample_rate,
        index_path=args.index_path,
        index_rate=args.index_rate,
        protect_f0=args.protect_f0
    )

def convert(
    model_path: str,
    input_file: str,
    output_file: str | None = None,
    pitch_shift_key: int = 0,
    f0_extractor_name: str = 'rmvpe',
    output_sr: int = 48000,
    index_path: str | None = None,
    index_rate: float = 0.75,
    protect_f0: float = 0.5
):
    print(f"Attempting to load model from: {model_path}")
    print(f"Attempting to load audio from: {input_file}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Audio file '{input_file}' not found.")

    # Check for index file if provided
    if index_path and not os.path.exists(index_path):
        print(f"Warning: Index file '{index_path}' not found. Conversion will proceed without index.")
        index_path = None # Set to None so VC class doesn't try to load it

    try:
        model_instance = loader(model_path)
        model_data = model_instance.model # This is the loaded .pth content
        if model_data is None:
            raise InterruptedError("Model not loaded successfully.")
        
        # Add the path to model_data if not already present, for VC class init
        if not hasattr(model_data, 'path'):
            model_data.path = model_path # Attach the path for the VC class

        print(f"Model loaded successfully from {model_path}")
        if isinstance(model_data, dict):
            print(f"Model keys: {model_data.keys()}")
            print(f"Model sample rate: {model_data.get('sample_rate', 'N/A')} Hz")
            print(f"Model hubert dim: {model_data.get('embedder_output_channels', 'N/A')}")
            print(f"Model n_speakers: {model_data.get('n_speakers', 'N/A')}")

        # Load input audio
        audio_input, input_sr = _load_audio_file(input_file) 
        print(f"Audio file loaded: {input_file} (Input SR: {input_sr} Hz, Duration: {len(audio_input)/input_sr:.2f} s)")

        # Perform Voice Conversion using the RVC pipeline
        converted_audio = _perform_vc(
            model_data=model_data, 
            audio=audio_input, 
            sr=input_sr,
            f0_extractor_name=f0_extractor_name,
            target_f0_shift=pitch_shift_key,
            output_sr=output_sr,
            index_path=index_path,
            index_rate=index_rate,
            protect_f0=protect_f0
        )

        # 3. Save the converted audio
        if output_file is None:
            date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            input_filename = os.path.splitext(os.path.basename(input_file))[0]
            model_filename = os.path.splitext(os.path.basename(model_path))[0]
            output_file = f"{input_filename}_to_{model_filename}_shifted{pitch_shift_key}k_{date_time}.wav"

        output_dir = os.path.join(os.getcwd(), "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        final_output_path = os.path.join(output_dir, output_file)
        
        sf.write(final_output_path, converted_audio, output_sr)
        print(f"Conversion complete! Output saved to: {final_output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except InterruptedError as e:
        print(f"Error during loading or conversion: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during RVC conversion: {e}")
        import traceback
        traceback.print_exc() 


if __name__ == '__main__':
    main()