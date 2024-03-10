"""
AlphaAgents Configuration Module

Handles environment variables, API keys, and model configuration.
"""

import os
import sys
from typing import Literal
from pydantic import BaseSettings, Field
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging(log_level="INFO"):
    logger.remove()
    logger.add(sys.stdout, level=log_level)
    logger.add("logs/alpha_agents.log", rotation="10 MB", level=log_level)
    return logger


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    
    # News API Configuration
    news_api_key: str = Field(default="", env="NEWS_API_KEY")
    
    # Risk Profile
    default_risk_profile: Literal["averse", "neutral"] = Field(
        default="neutral", 
        env="DEFAULT_RISK_PROFILE"
    )
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()


# LLM Configuration for AutoGen
def get_llm_config() -> dict:
    """Get LLM configuration for AutoGen agents."""
    settings = get_settings()
    return {
        "config_list": [
            {
                "model": settings.openai_model,
                "api_key": settings.openai_api_key,
            }
        ],
        "temperature": settings.openai_temperature,
        "max_tokens": settings.openai_max_tokens,
    }
