"""Model client implementations for different LLM providers."""

import time
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from .config import Config


@dataclass
class ModelResponse:
    """Response from a model."""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    latency: float = 0.0
    error: Optional[str] = None


class ModelClient(ABC):
    """Abstract base class for model clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Generate responses for multiple prompts."""
        pass


class NVIDIAClient(ModelClient):
    """Client for NVIDIA API."""
    
    def __init__(self, api_key: str, base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = "nvidia/nemotron-4-mini-hindi-4b"
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response using NVIDIA API."""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", 60)
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            latency = time.time() - start_time
            
            return ModelResponse(
                content=content,
                model=self.model,
                provider="nvidia",
                tokens_used=tokens_used,
                latency=latency
            )
        except requests.exceptions.RequestException as e:
            latency = time.time() - start_time
            return ModelResponse(
                content="",
                model=self.model,
                provider="nvidia",
                error=str(e),
                latency=latency
            )
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Generate responses for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class ZAIClient(ModelClient):
    """Client for ZAI API."""
    
    def __init__(self, api_key: str, model: str = "glm-4.7-flash", 
                 base_url: str = "https://api.z.ai/api/paas/v4"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response using ZAI API."""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", 60)
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            latency = time.time() - start_time
            
            return ModelResponse(
                content=content,
                model=self.model,
                provider="zai",
                tokens_used=tokens_used,
                latency=latency
            )
        except requests.exceptions.RequestException as e:
            latency = time.time() - start_time
            return ModelResponse(
                content="",
                model=self.model,
                provider="zai",
                error=str(e),
                latency=latency
            )
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Generate responses for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


def get_client(provider: str, config: Optional[Config] = None) -> ModelClient:
    """Factory function to get the appropriate model client."""
    if config is None:
        config = Config()
    
    if provider == "nvidia":
        return NVIDIAClient(
            api_key=config.nvidia_api_key,
            base_url=config.nvidia_base_url
        )
    elif provider == "zai":
        return ZAIClient(
            api_key=config.zai_api_key,
            model=config.zai_model,
            base_url=config.zai_base_url
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
