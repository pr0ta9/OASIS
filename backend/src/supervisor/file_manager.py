import os
from typing import Dict, List
from .state import MessagesState


class FileManager:
    """File classification and validation for multi-agent system."""
    
    def __init__(self):
        self.file_extensions = {
            'document': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx'],
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.ico'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus'],
            'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp']
        }
    
    def classify_uploaded_files(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Classify uploaded file paths into different types based on file extensions only.
        
        Args:
            file_paths: List of file paths from frontend uploads
            
        Returns:
            Dictionary with classified file paths:
            {
                'document_paths': [...],
                'image_paths': [...], 
                'audio_paths': [...],
                'video_paths': [...],
                'unrecognized_paths': [...]
            }
        """
        classified = {
            'document_paths': [],
            'image_paths': [],
            'audio_paths': [],
            'video_paths': [],
            'unrecognized_paths': []
        }
        
        for file_path in file_paths:
            if not file_path:
                continue
                
            # Get file extension
            _, ext = os.path.splitext(file_path.lower())
            
            # Classify by extension only
            file_classified = False
            for file_type, extensions in self.file_extensions.items():
                if ext in extensions:
                    classified[f'{file_type}_paths'].append(file_path)
                    file_classified = True
                    break
            
            # If extension not recognized, add to unrecognized
            if not file_classified:
                classified['unrecognized_paths'].append(file_path)
        
        return classified
    
    def validate_requirements(self, agent_type: str, state: MessagesState) -> Dict[str, any]:
        """
        Validate if required files exist for the specified agent type.
                 Args:
             agent_type: The agent type ('text_agent', 'image_agent', 'audio_agent', 'document_agent', 'video_agent')
            state: Current MessagesState with file paths
            
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'missing_file_type': str or None,
                'error_message': str or None,
                'available_files': List[str]
            }
        """
        # Map agent types to required file types and state fields
        agent_requirements = {
            'text_agent': {
                'required_files': None,  # Text agent doesn't require specific files
                'state_fields': []
            },
            'image_agent': {
                'required_files': 'image',
                'state_fields': ['image_paths']
            },
            'audio_agent': {
                'required_files': 'audio', 
                'state_fields': ['audio_paths']
            },
            'document_agent': {
                'required_files': 'document',
                'state_fields': ['document_paths']
            },
            'video_agent': {
                'required_files': 'video',
                'state_fields': ['video_paths']
            }
        }
        
        if agent_type not in agent_requirements:
            return {
                'valid': False,
                'missing_file_type': None,
                'error_message': f"Unknown agent type: {agent_type}",
                'available_files': []
            }
        
        requirements = agent_requirements[agent_type]
        
        # Text expert doesn't require specific files
        if requirements['required_files'] is None:
            return {
                'valid': True,
                'missing_file_type': None,
                'error_message': None,
                'available_files': []
            }
        
        # Check if required files exist
        available_files = []
        for field in requirements['state_fields']:
            if hasattr(state, field) and field in state:
                available_files.extend(state[field] or [])
        
        if not available_files:
            file_type = requirements['required_files']
            extensions = ', '.join(self.file_extensions[file_type])
            
            return {
                'valid': False,
                'missing_file_type': file_type,
                'error_message': (
                    f"❌ Cannot assign task to {agent_type.replace('_', ' ')} - no {file_type} files uploaded.\n\n"
                    f"Please upload a {file_type} file first:\n"
                    f"Supported formats: {extensions}\n\n"
                    f"Once you upload the {file_type} file, I'll be able to help you with your request."
                ),
                'available_files': []
            }
        
        return {
            'valid': True,
            'missing_file_type': None,
            'error_message': None,
            'available_files': available_files
        } 