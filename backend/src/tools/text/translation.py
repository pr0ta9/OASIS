from google.cloud import translate
from google.cloud import language_v1
from langchain_core.tools import tool
from typing import Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

def init_translation_client():
    """
    Initialize the Google Cloud Translation API client.
    
    This function creates the translation client only when called,
    avoiding automatic initialization on import.
    
    Returns:
        translate.Client: The initialized Google Cloud Translation client
        
    Raises:
        Exception: If client initialization fails
    """
    try:
        # Check for credentials
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")
        
        client = translate.Client()
        logger.info("✅ Google Cloud Translation API client initialized")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to initialize Translation API client: {e}")
        raise

def init_language_client():
    """
    Initialize the Google Cloud Natural Language API client.
    
    Returns:
        language_v1.LanguageServiceClient: The initialized Natural Language client
        
    Raises:
        Exception: If client initialization fails
    """
    try:
        # Check for credentials
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")
        
        client = language_v1.LanguageServiceClient()
        logger.info("✅ Google Cloud Natural Language API client initialized")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to initialize Natural Language API client: {e}")
        raise

@tool
def cloud_translation_tool(
    text: str,
    target_language: str,
    source_language: Optional[str] = None
) -> str:
    """
    Translate text using Google Cloud Translation API.
    
    This tool uses Google Cloud's powerful translation service to translate text
    between different languages with high accuracy.
    
    Args:
        text: The text to translate
        target_language: Target language code (e.g., 'es' for Spanish, 'fr' for French)
        source_language: Optional source language code. If not provided, will auto-detect
    
    Returns:
        Status message with translated text or error information
    """
    try:
        logger.info("🔄 Initializing Google Cloud Translation client...")
        client = init_translation_client()
    except Exception as e:
        return f"❌ Error: Failed to initialize Google Cloud Translation client: {str(e)}"
    
    try:
        # Perform translation
        if source_language:
            result = client.translate(
                text,
                source_language=source_language,
                target_language=target_language
            )
        else:
            result = client.translate(text, target_language=target_language)
        
        translated_text = result['translatedText']
        detected_language = result.get('detectedSourceLanguage', 'unknown')
        
        return f"✅ Translation successful!\n" \
               f"🔤 Original: {text}\n" \
               f"🌍 Detected language: {detected_language}\n" \
               f"📝 Translated ({target_language}): {translated_text}"
        
    except Exception as e:
        return f"❌ Error during translation: {str(e)}"

@tool
def cloud_language_analysis_tool(
    text: str,
    analysis_type: str = "all"
) -> str:
    """
    Analyze text using Google Cloud Natural Language API.
    
    This tool performs various text analysis tasks including sentiment analysis,
    entity extraction, and language detection.
    
    Args:
        text: The text to analyze
        analysis_type: Type of analysis ('sentiment', 'entities', 'syntax', 'all')
    
    Returns:
        Analysis results including sentiment, entities, and other insights
    """
    try:
        logger.info("🔄 Initializing Google Cloud Natural Language client...")
        client = init_language_client()
    except Exception as e:
        return f"❌ Error: Failed to initialize Google Cloud Natural Language client: {str(e)}"
    
    try:
        # Create document object
        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT
        )
        
        results = []
        
        # Sentiment analysis
        if analysis_type in ['sentiment', 'all']:
            sentiment_response = client.analyze_sentiment(
                request={'document': document}
            )
            sentiment = sentiment_response.document_sentiment
            results.append(f"😊 Sentiment: {sentiment.magnitude:.2f} magnitude, {sentiment.score:.2f} score")
        
        # Entity extraction
        if analysis_type in ['entities', 'all']:
            entities_response = client.analyze_entities(
                request={'document': document}
            )
            entities = entities_response.entities
            if entities:
                entity_list = []
                for entity in entities[:5]:  # Limit to top 5
                    entity_list.append(f"{entity.name} ({entity.type_.name})")
                results.append(f"🏷️ Entities: {', '.join(entity_list)}")
            else:
                results.append("🏷️ Entities: None detected")
        
        # Language detection
        if analysis_type in ['language', 'all']:
            # Language is automatically detected in the sentiment analysis
            results.append(f"🌍 Language: Auto-detected from content")
        
        return f"✅ Natural Language Analysis Results:\n" + "\n".join(results)
        
    except Exception as e:
        return f"❌ Error during text analysis: {str(e)}" 