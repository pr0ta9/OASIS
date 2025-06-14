from langchain_core.tools import tool
from typing import Optional, Literal, Dict, Any
import os
from pathlib import Path
from google.cloud import speech


def init_speech_client():
    """
    Initialize the Google Cloud Speech-to-Text client.
    
    This function creates a Speech client only when called,
    avoiding automatic initialization on import.
    
    Returns:
        client: The initialized Google Cloud Speech client
        
    Raises:
        Exception: If client initialization fails
    """
    try:
        client = speech.SpeechClient()
        print("✅ Google Cloud Speech-to-Text client initialized")
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize Speech client: {str(e)}")


def speech_to_text(
    config: 'speech.RecognitionConfig',
    audio: 'speech.RecognitionAudio',
) -> 'speech.RecognizeResponse':
    """
    Core function to perform speech-to-text transcription.
    
    Args:
        config: Recognition configuration
        audio: Audio data to be recognized
        
    Returns:
        Response containing transcription results
    """
    client = init_speech_client()
    response = client.recognize(config=config, audio=audio)
    return response


@tool
def speech_to_text_tool(
    audio_file_path: str,
    language_code: str = "en-US",
    enable_automatic_punctuation: bool = True,
    enable_word_time_offsets: bool = False,
    audio_encoding: Optional[Literal["flac", "wav", "mp3", "ogg", "webm"]] = None,
    sample_rate_hertz: Optional[int] = None
) -> str:
    """
    Transcribe speech from an audio file using Google Cloud Speech-to-Text API.
    
    This tool converts audio files to text with language detection and confidence scoring.
    Supports local files and Google Cloud Storage URIs.
    
    Args:
        audio_file_path: Path to the audio file or GCS URI (gs://bucket/file)
        language_code: Language code for transcription (e.g., "en-US", "es-ES", "auto" for detection)
        enable_automatic_punctuation: Whether to enable automatic punctuation
        enable_word_time_offsets: Whether to include word-level timestamps
        audio_encoding: Audio encoding format (auto-detected if None)
        sample_rate_hertz: Sample rate of the audio (auto-detected if None)
    
    Returns:
        JSON string containing transcript, confidence, detected language, and metadata
    """
    
    try:
        # Determine if it's a GCS URI or local file
        is_gcs_uri = audio_file_path.startswith("gs://")
        
        if not is_gcs_uri:
            # Check if local file exists
            if not os.path.exists(audio_file_path):
                return f"❌ Error: Audio file not found: {audio_file_path}"
        
        # Auto-detect encoding from file extension if not provided
        if audio_encoding is None:
            file_ext = Path(audio_file_path).suffix.lower().lstrip('.')
            encoding_map = {
                'flac': speech.RecognitionConfig.AudioEncoding.FLAC,
                'wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
                'mp3': speech.RecognitionConfig.AudioEncoding.MP3,
                'ogg': speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
                'webm': speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            }
            detected_encoding = encoding_map.get(file_ext)
            if detected_encoding is None:
                return f"❌ Error: Unsupported audio format: {file_ext}. Supported: flac, wav, mp3, ogg, webm"
        else:
            encoding_map = {
                'flac': speech.RecognitionConfig.AudioEncoding.FLAC,
                'wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
                'mp3': speech.RecognitionConfig.AudioEncoding.MP3,
                'ogg': speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
                'webm': speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            }
            detected_encoding = encoding_map[audio_encoding]
        
        # Create recognition config
        config_params = {
            "encoding": detected_encoding,
            "enable_automatic_punctuation": enable_automatic_punctuation,
            "enable_word_time_offsets": enable_word_time_offsets,
        }
        
        # Handle language code
        if language_code.lower() == "auto":
            # Enable automatic language detection
            config_params["language_code"] = "en-US"  # Fallback
            config_params["alternative_language_codes"] = ["es-ES", "fr-FR", "de-DE", "ja-JP"]
        else:
            config_params["language_code"] = language_code
        
        # Add sample rate if provided
        if sample_rate_hertz:
            config_params["sample_rate_hertz"] = sample_rate_hertz
        
        config = speech.RecognitionConfig(**config_params)
        
        # Create audio object
        if is_gcs_uri:
            audio = speech.RecognitionAudio(uri=audio_file_path)
        else:
            with open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
        
        # Perform transcription
        print(f"🔄 Transcribing audio: {audio_file_path}")
        response = speech_to_text(config, audio)
        
        # Process results
        if not response.results:
            return "⚠️ No speech detected in the audio file."
        
        results = []
        for i, result in enumerate(response.results):
            best_alternative = result.alternatives[0]
            
            result_data = {
                "segment": i + 1,
                "transcript": best_alternative.transcript,
                "confidence": f"{best_alternative.confidence:.0%}" if best_alternative.confidence else "N/A",
                "language_code": result.language_code if hasattr(result, 'language_code') else language_code
            }
            
            # Add word-level timestamps if enabled
            if enable_word_time_offsets and hasattr(best_alternative, 'words'):
                words_with_times = []
                for word_info in best_alternative.words:
                    words_with_times.append({
                        "word": word_info.word,
                        "start_time": f"{word_info.start_time.total_seconds():.2f}s",
                        "end_time": f"{word_info.end_time.total_seconds():.2f}s"
                    })
                result_data["word_timestamps"] = words_with_times
            
            results.append(result_data)
        
        # Format output
        full_transcript = " ".join([r["transcript"] for r in results])
        avg_confidence = sum([float(r["confidence"].rstrip('%')) for r in results if r["confidence"] != "N/A"]) / len(results)
        detected_language = results[0]["language_code"] if results else language_code
        
        output = {
            "success": True,
            "full_transcript": full_transcript,
            "detected_language": detected_language,
            "average_confidence": f"{avg_confidence:.0f}%",
            "segments": len(results),
            "detailed_results": results if len(results) > 1 else None
        }
        
        # Create readable summary
        summary = f"✅ Audio transcribed successfully!\n"
        summary += f"📝 Transcript: {full_transcript}\n"
        summary += f"🌍 Language: {detected_language}\n"
        summary += f"📊 Confidence: {avg_confidence:.0f}%\n"
        summary += f"📋 Segments: {len(results)}"
        
        return summary
        
    except FileNotFoundError:
        return f"❌ Error: Audio file not found: {audio_file_path}"
    except Exception as e:
        return f"❌ Error during transcription: {str(e)}"


 