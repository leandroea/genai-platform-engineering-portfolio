"""Configuration management for the LLM Evaluation Framework."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: str
    api_key: str
    base_url: Optional[str] = None


@dataclass
class Config:
    """Main configuration for the evaluation framework."""
    
    # NVIDIA Configuration
    nvidia_api_key: Optional[str] = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    
    # ZAI Configuration
    zai_api_key: Optional[str] = None
    zai_base_url: str = "https://api.z.ai/api/paas/v4"
    zai_model: str = "glm-4.7-flash"
    
    # Default models to evaluate
    default_models: list = None
    
    # Evaluation settings
    max_retries: int = 3
    timeout: int = 60
    
    def __post_init__(self):
        """Load environment variables and set defaults."""
        load_dotenv()
        
        # Load NVIDIA config
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY") or self.nvidia_api_key
        self.nvidia_base_url = os.getenv("NVIDIA_BASE_URL") or self.nvidia_base_url
        
        # Load ZAI config
        self.zai_api_key = os.getenv("ZAI_API_KEY") or self.zai_api_key
        self.zai_base_url = os.getenv("ZAI_BASE_URL") or self.zai_base_url
        self.zai_model = os.getenv("ZAI_MODEL") or self.zai_model
        
        # Set default models
        if self.default_models is None:
            self.default_models = [
                ModelConfig(
                    name="nvidia/nemotron-4-mini-hindi-4b",
                    provider="nvidia",
                    api_key=self.nvidia_api_key,
                    base_url=self.nvidia_base_url
                ),
                ModelConfig(
                    name=self.zai_model,
                    provider="zai",
                    api_key=self.zai_api_key,
                    base_url=self.zai_base_url
                )
            ]
    
    def get_model_config(self, provider: str) -> ModelConfig:
        """Get model configuration for a specific provider."""
        if provider == "nvidia":
            return ModelConfig(
                name="nvidia/nemotron-4-mini-hindi-4b",
                provider="nvidia",
                api_key=self.nvidia_api_key,
                base_url=self.nvidia_base_url
            )
        elif provider == "zai":
            return ModelConfig(
                name=self.zai_model,
                provider="zai",
                api_key=self.zai_api_key,
                base_url=self.zai_base_url
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.nvidia_api_key and not self.zai_api_key:
            raise ValueError("At least one API key must be configured")
        return True
