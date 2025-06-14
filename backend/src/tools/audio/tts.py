from langchain_core.tools import tool
from typing import Optional, Literal
import os
from pathlib import Path

# Optional imports with fallback
try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Google Cloud Text-to-Speech not available: {e}")
    GOOGLE_TTS_AVAILABLE = False
    texttospeech = None


def init_tts_client():
    """
    Initialize the Google Cloud Text-to-Speech client.
    
    This function creates a TTS client only when called,
    avoiding automatic initialization on import.
    
    Returns:
        client: The initialized Google Cloud TTS client
        
    Raises:
        Exception: If client initialization fails
    """
    if not GOOGLE_TTS_AVAILABLE:
        raise Exception("Google Cloud Text-to-Speech is not available. Please install google-cloud-texttospeech and set up authentication.")
    
    try:
        client = texttospeech.TextToSpeechClient()
        print("✅ Google Cloud Text-to-Speech client initialized")
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize TTS client: {str(e)}")


@tool
def text_to_speech_tool(
    text: str,
    output_path: Optional[str] = None,
    audio_format: Literal["mp3", "wav", "ogg"] = "mp3",
    voice_gender: Literal["male", "female", "neutral"] = "female",
    language_code: str = "en-US",
    voice_name: Optional[str] = None
) -> str:
    """
    Convert text to speech using Google Cloud Text-to-Speech API.
    
    This tool converts input text to audio in various formats with customizable voice options.
    
    Args:
        text: The text to convert to speech
        output_path: Optional path where the audio will be saved. 
                    If not provided, will use 'tts_output' with appropriate extension
        audio_format: Audio format for output ("mp3", "wav", "ogg")
        voice_gender: Voice gender preference ("male", "female", "neutral")
        language_code: Language code (e.g., "en-US", "en-GB", "es-ES")
        voice_name: Optional specific voice name. If not provided, will auto-select based on gender
    
    Returns:
        Status message indicating success or failure of the text-to-speech conversion
    """
    if not GOOGLE_TTS_AVAILABLE:
        return "❌ Error: Google Cloud Text-to-Speech is not available. " \
               "Please install google-cloud-texttospeech and set up authentication."
    
    # Initialize TTS client
    try:
        print("🔄 Initializing TTS client...")
        client = init_tts_client()
    except Exception as e:
        return f"❌ Error: Failed to initialize TTS client: {str(e)}"
    
    try:
        # Generate output path if not provided
        if output_path is None:
            output_path = f"tts_output.{audio_format}"
        else:
            # Ensure the output path has the correct extension
            output_path = Path(output_path)
            output_path = output_path.with_suffix(f".{audio_format}")
            output_path = str(output_path)
        
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Map gender to Google TTS enum
        gender_mapping = {
            "male": texttospeech.SsmlVoiceGender.MALE,
            "female": texttospeech.SsmlVoiceGender.FEMALE,
            "neutral": texttospeech.SsmlVoiceGender.NEUTRAL
        }
        
        # Auto-select voice name if not provided
        if voice_name is None:
            if language_code == "en-US":
                voice_name = {
                    "female": "en-US-Neural2-A",
                    "male": "en-US-Neural2-D", 
                    "neutral": "en-US-Neural2-C"
                }.get(voice_gender, "en-US-Neural2-A")
            else:
                # For other languages, let Google auto-select
                voice_name = None
        
        # Build the voice request
        voice_params = {
            "language_code": language_code,
            "ssml_gender": gender_mapping[voice_gender]
        }
        if voice_name:
            voice_params["name"] = voice_name
            
        voice = texttospeech.VoiceSelectionParams(**voice_params)
        
        # Map audio format to Google TTS enum
        format_mapping = {
            "mp3": texttospeech.AudioEncoding.MP3,
            "wav": texttospeech.AudioEncoding.LINEAR16,
            "ogg": texttospeech.AudioEncoding.OGG_OPUS
        }
        
        # Select the type of audio file
        audio_config = texttospeech.AudioConfig(
            audio_encoding=format_mapping[audio_format]
        )
        
        # Perform the text-to-speech request
        print(f"🔄 Converting text to speech...")
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save the output to file
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        
        return f"✅ Text successfully converted to speech! Audio saved to: {output_path}"
        
    except FileNotFoundError as e:
        return f"❌ Error: Output directory not found: {str(e)}"
    except Exception as e:
        return f"❌ Error during text-to-speech conversion: {str(e)}"


@tool 
def list_available_voices(language_code: str = "en-US") -> str:
    """
    List available voices for a given language code.
    
    Args:
        language_code: Language code to list voices for (e.g., "en-US", "en-GB")
    
    Returns:
        String containing formatted list of available voices
    """
    if not GOOGLE_TTS_AVAILABLE:
        return "❌ Error: Google Cloud Text-to-Speech is not available."
    
    try:
        client = init_tts_client()
        response = client.list_voices(language_code=language_code)
        
        voices_info = []
        for voice in response.voices:
            voices_info.append(
                f"Name: {voice.name}, "
                f"Language(s): {', '.join(voice.language_codes)}, "
                f"Gender: {voice.ssml_gender.name}"
            )
        
        if voices_info:
            return f"Available voices for {language_code}:\n" + "\n".join(voices_info)
        else:
            return f"No voices found for language code: {language_code}"
            
    except Exception as e:
        return f"❌ Error listing voices: {str(e)}" 