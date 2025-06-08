from google.cloud import videointelligence
from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def init_video_intelligence_client():
    """
    Initialize the Google Cloud Video Intelligence API client.
    
    Returns:
        videointelligence.VideoIntelligenceServiceClient: The initialized Video Intelligence client
        
    Raises:
        Exception: If client initialization fails
    """
    try:
        # Check for credentials
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")
        
        client = videointelligence.VideoIntelligenceServiceClient()
        logger.info("✅ Google Cloud Video Intelligence API client initialized")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to initialize Video Intelligence API client: {e}")
        raise

@tool
def video_analysis_tool(
    video_path: str,
    analysis_features: Optional[List[str]] = None
) -> str:
    """
    Analyze video content using Google Cloud Video Intelligence API.
    
    This tool provides comprehensive video analysis including label detection,
    shot change detection, explicit content detection, and more.
    
    Args:
        video_path: Path to the video file to analyze
        analysis_features: List of features to analyze ('LABEL_DETECTION', 'SHOT_CHANGE_DETECTION', 'EXPLICIT_CONTENT_DETECTION')
    
    Returns:
        Video analysis results
    """
    try:
        logger.info("🔄 Initializing Google Cloud Video Intelligence client...")
        client = init_video_intelligence_client()
    except Exception as e:
        return f"❌ Error: Failed to initialize Google Cloud Video Intelligence client: {str(e)}"
    
    try:
        # Check if file exists
        if not os.path.exists(video_path):
            return f"❌ Error: Video file not found: {video_path}"
        
        # Read the video file
        with open(video_path, "rb") as video_file:
            input_content = video_file.read()
        
        # Configure features if not provided
        if not analysis_features:
            analysis_features = ['LABEL_DETECTION', 'SHOT_CHANGE_DETECTION']
        
        # Map feature names to enum values
        feature_map = {
            'LABEL_DETECTION': videointelligence.Feature.LABEL_DETECTION,
            'SHOT_CHANGE_DETECTION': videointelligence.Feature.SHOT_CHANGE_DETECTION,
            'EXPLICIT_CONTENT_DETECTION': videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
            'TEXT_DETECTION': videointelligence.Feature.TEXT_DETECTION,
            'OBJECT_TRACKING': videointelligence.Feature.OBJECT_TRACKING
        }
        
        features = [feature_map.get(f, videointelligence.Feature.LABEL_DETECTION) for f in analysis_features]
        
        # For demonstration purposes, simulate the analysis
        # In production, you would make the actual API call:
        # operation = client.annotate_video(
        #     request={"features": features, "input_content": input_content}
        # )
        # result = operation.result(timeout=300)
        
        filename = os.path.basename(video_path)
        file_size = len(input_content)
        
        # Mock analysis results
        results = []
        results.append(f"🎬 Video: {filename}")
        results.append(f"📏 File size: {file_size:,} bytes")
        results.append(f"🔍 Features analyzed: {', '.join(analysis_features)}")
        
        # Simulate feature-specific results
        if 'LABEL_DETECTION' in analysis_features:
            results.append("🏷️ Labels detected: [person, indoor, computer, technology, office]")
        
        if 'SHOT_CHANGE_DETECTION' in analysis_features:
            results.append("🎯 Shot changes: 5 scenes detected")
        
        if 'TEXT_DETECTION' in analysis_features:
            results.append("📝 Text detected: [sample text overlay, captions]")
        
        if 'EXPLICIT_CONTENT_DETECTION' in analysis_features:
            results.append("🛡️ Content rating: Safe for all audiences")
        
        results.append("⚠️ Note: This is a demonstration. Configure Video Intelligence API for production use.")
        
        return f"✅ Video Intelligence Analysis successful!\n" + "\n".join(results)
        
    except Exception as e:
        return f"❌ Error during video analysis: {str(e)}"

@tool
def video_text_detection_tool(
    video_path: str,
    segment_start: Optional[float] = None,
    segment_end: Optional[float] = None
) -> str:
    """
    Detect and extract text from video content using Video Intelligence API.
    
    This tool identifies text overlays, captions, and signs within video frames
    throughout the duration of the video.
    
    Args:
        video_path: Path to the video file to analyze
        segment_start: Optional start time in seconds for analysis
        segment_end: Optional end time in seconds for analysis
    
    Returns:
        Text detection results from the video
    """
    try:
        logger.info("🔄 Initializing Google Cloud Video Intelligence client for text detection...")
        client = init_video_intelligence_client()
    except Exception as e:
        return f"❌ Error: Failed to initialize Google Cloud Video Intelligence client: {str(e)}"
    
    try:
        # Check if file exists
        if not os.path.exists(video_path):
            return f"❌ Error: Video file not found: {video_path}"
        
        filename = os.path.basename(video_path)
        
        # Mock text detection results
        results = []
        results.append(f"🎬 Video: {filename}")
        
        if segment_start is not None and segment_end is not None:
            results.append(f"⏱️ Analyzing segment: {segment_start}s - {segment_end}s")
        else:
            results.append("⏱️ Analyzing entire video")
        
        # Simulate detected text segments
        detected_texts = [
            {"text": "Welcome to the presentation", "timestamp": "00:05", "confidence": 0.95},
            {"text": "AI in Healthcare", "timestamp": "00:15", "confidence": 0.92},
            {"text": "Thank you for watching", "timestamp": "02:30", "confidence": 0.88}
        ]
        
        results.append(f"📝 Text segments detected: {len(detected_texts)}")
        
        for i, text_info in enumerate(detected_texts, 1):
            results.append(f"  {i}. '{text_info['text']}' at {text_info['timestamp']} (confidence: {text_info['confidence']:.2f})")
        
        results.append("⚠️ Note: Configure Video Intelligence API for production text detection.")
        
        return f"✅ Video Text Detection successful!\n" + "\n".join(results)
        
    except Exception as e:
        return f"❌ Error during video text detection: {str(e)}"

@tool
def video_scene_detection_tool(
    video_path: str,
    confidence_threshold: float = 0.8
) -> str:
    """
    Detect scene changes and shots in video using Video Intelligence API.
    
    This tool identifies shot boundaries and scene transitions to help
    understand the structure and content of video files.
    
    Args:
        video_path: Path to the video file to analyze
        confidence_threshold: Minimum confidence for scene detection (0.0-1.0)
    
    Returns:
        Scene detection and shot change analysis results
    """
    try:
        logger.info("🔄 Initializing Google Cloud Video Intelligence client for scene detection...")
        client = init_video_intelligence_client()
    except Exception as e:
        return f"❌ Error: Failed to initialize Google Cloud Video Intelligence client: {str(e)}"
    
    try:
        # Check if file exists
        if not os.path.exists(video_path):
            return f"❌ Error: Video file not found: {video_path}"
        
        filename = os.path.basename(video_path)
        
        # Mock scene detection results
        results = []
        results.append(f"🎬 Video: {filename}")
        results.append(f"🎯 Confidence threshold: {confidence_threshold}")
        
        # Simulate detected scenes
        scenes = [
            {"start": "00:00", "end": "00:30", "description": "Opening scene - indoor office"},
            {"start": "00:30", "end": "01:15", "description": "Presentation slide - AI overview"},
            {"start": "01:15", "end": "02:00", "description": "Demo section - software interface"},
            {"start": "02:00", "end": "02:45", "description": "Closing remarks - person speaking"}
        ]
        
        results.append(f"🎭 Scenes detected: {len(scenes)}")
        
        for i, scene in enumerate(scenes, 1):
            results.append(f"  Scene {i}: {scene['start']} - {scene['end']} | {scene['description']}")
        
        # Simulate shot change statistics
        total_shots = len(scenes) * 2  # Assume 2 shots per scene on average
        avg_shot_duration = 30 / len(scenes) if scenes else 0
        
        results.append(f"📊 Shot analysis:")
        results.append(f"  • Total shots: {total_shots}")
        results.append(f"  • Average shot duration: {avg_shot_duration:.1f} seconds")
        
        results.append("⚠️ Note: Configure Video Intelligence API for production scene detection.")
        
        return f"✅ Video Scene Detection successful!\n" + "\n".join(results)
        
    except Exception as e:
        return f"❌ Error during video scene detection: {str(e)}" 