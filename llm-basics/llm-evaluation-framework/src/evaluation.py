"""Main evaluation runner for LLM models."""

from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm

from .config import Config
from .models import ModelClient, get_client, ModelResponse
from .dataset import Dataset
from .metrics import MetricsCalculator


class EvaluationResult:
    """Container for evaluation results."""
    
    def __init__(self, model_name: str, provider: str):
        self.model_name = model_name
        self.provider = provider
        self.responses: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
    
    def add_response(self, prompt: str, response: ModelResponse):
        """Add a model response to the results."""
        self.responses.append({
            "prompt": prompt,
            "response": response.content,
            "model": response.model,
            "provider": response.provider,
            "tokens_used": response.tokens_used,
            "latency": response.latency,
            "error": response.error
        })
    
    def calculate_metrics(self, references: List[str], 
                          ground_truths: Optional[List[str]] = None):
        """Calculate metrics for all responses."""
        predictions = [r["response"] for r in self.responses]
        
        self.metrics = MetricsCalculator.calculate_all(
            predictions, references, ground_truths
        )
        
        # Add latency and token stats
        self.metrics.update(MetricsCalculator.calculate_latency_stats(self.responses))
        self.metrics.update(MetricsCalculator.calculate_token_stats(self.responses))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "responses": self.responses,
            "metrics": self.metrics
        }
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.responses)


class Evaluator:
    """Main evaluator class for running LLM evaluations."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.results: Dict[str, EvaluationResult] = {}
    
    def evaluate_model(self, provider: str, dataset: Dataset, 
                       show_progress: bool = True) -> EvaluationResult:
        """
        Evaluate a model on a dataset.
        
        Args:
            provider: Model provider ("nvidia" or "zai")
            dataset: Dataset to evaluate on
            show_progress: Whether to show progress bar
            
        Returns:
            EvaluationResult object
        """
        client = get_client(provider, self.config)
        
        model_config = self.config.get_model_config(provider)
        result = EvaluationResult(model_config.name, provider)
        
        prompts = dataset.get_prompts()
        
        iterator = tqdm(prompts, desc=f"Evaluating {provider}") if show_progress else prompts
        
        for prompt in iterator:
            response = client.generate(prompt)
            result.add_response(prompt, response)
        
        # Calculate metrics
        references = dataset.get_references()
        ground_truths = dataset.get_ground_truths()
        
        # Filter out empty ground truths
        valid_indices = [i for i, gt in enumerate(ground_truths) if gt]
        if valid_indices:
            valid_references = [references[i] for i in valid_indices]
            valid_ground_truths = [ground_truths[i] for i in valid_indices]
            result.calculate_metrics(valid_references, valid_ground_truths)
        else:
            result.calculate_metrics(references)
        
        self.results[provider] = result
        return result
    
    def evaluate_models(self, providers: List[str], dataset: Dataset,
                        show_progress: bool = True) -> Dict[str, EvaluationResult]:
        """
        Evaluate multiple models on a dataset.
        
        Args:
            providers: List of model providers
            dataset: Dataset to evaluate on
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary of EvaluationResult objects keyed by provider
        """
        for provider in providers:
            self.evaluate_model(provider, dataset, show_progress)
        
        return self.results
    
    def get_results(self, provider: Optional[str] = None) -> Dict[str, EvaluationResult]:
        """Get evaluation results."""
        if provider:
            return {provider: self.results[provider]}
        return self.results
    
    def save_results(self, output_path: str):
        """Save evaluation results to file."""
        import json
        
        results_dict = {
            provider: result.to_dict()
            for provider, result in self.results.items()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all evaluation results."""
        summary = {}
        
        for provider, result in self.results.items():
            summary[provider] = {
                "model_name": result.model_name,
                "metrics": result.metrics
            }
        
        return summary
