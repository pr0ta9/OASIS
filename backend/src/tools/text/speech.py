from google.cloud import speech
from google.cloud import texttospeech
from langchain_core.tools import tool
from typing import Optional
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def init_speech_clients():
    """
    Initialize the Google Cloud Speech-to-Text and Text-to-Speech clients.
    
    Returns:
        tuple: (speech_client, tts_client) - The initialized clients
        
    Raises:
        Exception: If client initialization fails
    """
    try:
        # Check for credentials
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")
        
        speech_client = speech.SpeechClient()
        tts_client = texttospeech.TextToSpeechClient()
        
        logger.info("✅ Google Cloud Speech API clients initialized")
        return speech_client, tts_client
    except Exception as e:
        logger.error(f"❌ Failed to initialize Speech API clients: {e}")
        raise

@tool
def speech_to_text_tool(
    audio_path: str,
    language_code: str = "en-US",
    audio_encoding: str = "LINEAR16"
) -> str:
    """
    Transcribe audio to text using Google Cloud Speech-to-Text API.
    
    This tool converts audio files to text with high accuracy and supports
    multiple languages and audio formats.
    
    Args:
        audio_path: Path to the audio file to transcribe
        language_code: Language code for transcription (e.g., 'en-US', 'es-ES')
        audio_encoding: Audio encoding format ('LINEAR16', 'FLAC', 'MP3', 'WEBM_OPUS')
    
    Returns:
        Transcribed text or error message
    """
    try:
        logger.info("🔄 Initializing Google Cloud Speech client...")
        speech_client, _ = init_speech_clients()
    except Exception as e:
        return f"❌ Error: Failed to initialize Google Cloud Speech client: {str(e)}"
    
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            return f"❌ Error: Audio file not found: {audio_path}"
        
        # Read audio file
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()
        
        # Configure audio settings
        encoding_map = {
            "LINEAR16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "FLAC": speech.RecognitionConfig.AudioEncoding.FLAC,
            "MP3": speech.RecognitionConfig.AudioEncoding.MP3,
            "WEBM_OPUS": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
        }
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=encoding_map.get(audio_encoding, speech.RecognitionConfig.AudioEncoding.LINEAR16),
            sample_rate_hertz=16000,
            language_code=language_code,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True
        )
        
        # Perform transcription
        response = speech_client.recognize(config=config, audio=audio)
        
        if not response.results:
            return f"⚠️ No speech detected in audio file: {audio_path}"
        
        # Extract transcribed text
        transcripts = []
        total_confidence = 0
        
        for result in response.results:
            alternative = result.alternatives[0]
            transcripts.append(alternative.transcript)
            total_confidence += alternative.confidence
        
        full_transcript = " ".join(transcripts)
        avg_confidence = total_confidence / len(response.results) if response.results else 0
        
        return f"✅ Speech-to-Text successful!\n" \
               f"🎤 Audio file: {os.path.basename(audio_path)}\n" \
               f"🌍 Language: {language_code}\n" \
               f"🎯 Confidence: {avg_confidence:.2f}\n" \
               f"📝 Transcript: {full_transcript}"
        
    except Exception as e:
        return f"❌ Error during speech-to-text conversion: {str(e)}"

@tool
def text_to_speech_tool(
    text: str,
    language_code: str = "en-US",
    voice_name: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Convert text to speech using Google Cloud Text-to-Speech API.
    
    This tool generates natural-sounding speech from text input with various
    voice options and languages.
    
    Args:
        text: The text to convert to speech
        language_code: Language code for speech (e.g., 'en-US', 'es-ES')
        voice_name: Specific voice name (optional, will use default for language)
        output_path: Path to save the audio file (optional, will auto-generate)
    
    Returns:
        Status message with output file path or error information
    """
    try:
        logger.info("🔄 Initializing Google Cloud Text-to-Speech client...")
        _, tts_client = init_speech_clients()
    except Exception as e:
        return f"❌ Error: Failed to initialize Google Cloud Text-to-Speech client: {str(e)}"
    
    try:
        # Set up synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Configure voice settings
        if voice_name:
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )
        else:
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
        
        # Configure audio format
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        # Perform text-to-speech
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Generate output path if not provided
        if output_path is None:
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            output_path = f"tts_output_{text_hash}.mp3"
        
        # Save audio content to file
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        
        return f"✅ Text-to-Speech successful!\n" \
               f"📝 Text: {text[:50]}{'...' if len(text) > 50 else ''}\n" \
               f"🌍 Language: {language_code}\n" \
               f"🎵 Voice: {voice_name or 'Default'}\n" \
               f"🔊 Audio saved to: {output_path}"
        
    except Exception as e:
        return f"❌ Error during text-to-speech conversion: {str(e)}" 