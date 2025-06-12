from langchain_core.tools import tool
from typing import Optional

# Optional imports with fallback
try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    MODELSCOPE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ModelScope not available: {e}")
    MODELSCOPE_AVAILABLE = False
    pipeline = None
    Tasks = None


def init_denoise():
    """
    Initialize the ModelScope denoise model.
    
    This function loads the speech enhancement model only when called,
    avoiding automatic initialization on import.
    
    Returns:
        pipeline: The initialized ModelScope pipeline
        
    Raises:
        Exception: If model initialization fails
    """
    if not MODELSCOPE_AVAILABLE:
        raise Exception("ModelScope is not available. Please install modelscope and resolve dependency conflicts.")
    
    # ModelScope speech enhancement model path
    model_path = 'damo/speech_zipenhancer_ans_multiloss_16k_base'
    # Alternative model: 'damo/speech_frcrn_ans_cirm_16k'
    
    denoise_model = pipeline(
        Tasks.acoustic_noise_suppression,
        model=model_path
    )
    print(f"✅ ModelScope denoise model loaded: {model_path}")
    return denoise_model


@tool
def audio_denoise_tool(
    input_path: str, 
    output_path: Optional[str] = None,
) -> str:
    """
    Denoise an audio file to remove background noise and enhance speech quality.
    
    This tool uses ModelScope's speech enhancement model to clean up audio files.
    Supports common audio formats like WAV, MP3, etc.
    
    Args:
        input_path: Path to the input audio file that needs denoising
        output_path: Optional path where the cleaned audio will be saved. 
                    If not provided, will use input filename with '_denoised' suffix
    
    Returns:
        Status message indicating success or failure of the denoising process
    """
    if not MODELSCOPE_AVAILABLE:
        return "❌ Error: ModelScope audio denoising is not available due to dependency conflicts. " \
               "Please resolve the ModelScope/datasets library compatibility issue."
    
    # Initialize model 
    try:
        print("🔄 Initializing denoise model...")
        denoise_model = init_denoise()
    except Exception as e:
        return f"❌ Error: Failed to initialize ModelScope denoise model: {str(e)}"
    
    try:
        # Generate output path if not provided
        if output_path is None:
            from pathlib import Path
            input_file = Path(input_path)
            output_path = str(input_file.parent / f"{input_file.stem}_denoised{input_file.suffix}")
        
        # Process the audio file
        result = denoise_model(input_path, output_path=output_path)
        
        if result and "output_pcm" in result:
            return f"✅ Audio successfully denoised! Output saved to: {output_path}"
        else:
            return f"⚠️ Denoising completed but result format unexpected. Check: {output_path}"
            
    except FileNotFoundError:
        return f"❌ Error: Input file not found: {input_path}"
    except Exception as e:
        return f"❌ Error during denoising: {str(e)}"


