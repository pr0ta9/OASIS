"""
Settings and configuration management for OASIS.
"""
import os
from typing import Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Google AI Configuration
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    google_project_id: Optional[str] = Field(default=None, env="GOOGLE_PROJECT_ID")
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_organization: Optional[str] = Field(default=None, env="OPENAI_ORGANIZATION")
    
    # MongoDB Configuration
    mongo_uri: str = Field(default="mongodb://localhost:27017", env="MONGO_URI")
    
    # Application Configuration
    app_mode: str = Field(default="simple", env="APP_MODE")  # simple or developer
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Local Model Configuration
    local_model_path: Path = Field(default=Path("./models"), env="LOCAL_MODEL_PATH")
    cache_size_gb: int = Field(default=10, env="CACHE_SIZE_GB")
    
    # UI Configuration
    theme: str = Field(default="dark", env="THEME")  # light or dark
    window_size: str = Field(default="1024x768", env="WINDOW_SIZE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def is_google_configured(self) -> bool:
        """Check if Google AI is properly configured."""
        return bool(self.google_api_key)
    
    def is_openai_configured(self) -> bool:
        """Check if OpenAI is properly configured."""
        return bool(self.openai_api_key)
    
    def get_window_dimensions(self) -> tuple[int, int]:
        """Parse window size string into width and height."""
        try:
            width, height = self.window_size.split('x')
            return int(width), int(height)
        except:
            return 1024, 768

# Global settings instance
settings = Settings() 