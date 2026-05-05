"""Model comparison utilities."""

import pandas as pd
from typing import Dict, Any, List, Optional
from .evaluation import EvaluationResult


class ModelComparator:
    """Compare evaluation results across different models."""
    
    def __init__(self, results: Dict[str, EvaluationResult]):
        """
        Initialize comparator.
        
        Args:
            results: Dictionary of EvaluationResult objects
        """
        self.results = results
    
    def compare_metrics(self, metric_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare metrics across models.
        
        Args:
            metric_names: List of metric names to compare. 
                         If None, compares all available metrics.
                         
        Returns:
            DataFrame with metrics comparison
        """
        comparison = {}
        
        for provider, result in self.results.items():
            metrics = result.metrics.copy()
            
            if metric_names:
                metrics = {k: v for k, v in metrics.items() if k in metric_names}
            
            comparison[provider] = metrics
        
        df = pd.DataFrame(comparison).T
        return df
    
    def get_winner(self, metric: str) -> str:
        """
        Get the winning model for a specific metric.
        
        Args:
            metric: Metric name to compare
            
        Returns:
            Provider name of the winning model
        """
        scores = {}
        
        for provider, result in self.results.items():
            if metric in result.metrics:
                scores[provider] = result.metrics[metric]
        
        if not scores:
            return None
        
        # For latency metrics, lower is better
        if "latency" in metric.lower():
            return min(scores, key=scores.get)
        
        # For all other metrics, higher is better
        return max(scores, key=scores.get)
    
    def rank_models(self, metric: str, ascending: bool = False) -> List[tuple]:
        """
        Rank models by a specific metric.
        
        Args:
            metric: Metric name to rank by
            ascending: If True, sort ascending (for latency)
            
        Returns:
            List of (provider, score) tuples sorted by metric
        """
        scores = []
        
        for provider, result in self.results.items():
            if metric in result.metrics:
                scores.append((provider, result.metrics[metric]))
        
        return sorted(scores, key=lambda x: x[1], reverse=not ascending)
    
    def generate_report(self) -> str:
        """
        Generate a text comparison report.
        
        Returns:
            Formatted comparison report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("MODEL COMPARISON REPORT")
        lines.append("=" * 60)
        
        # Compare each metric
        all_metrics = set()
        for result in self.results.values():
            all_metrics.update(result.metrics.keys())
        
        for metric in sorted(all_metrics):
            lines.append(f"\n{metric.upper()}:")
            lines.append("-" * 40)
            
            scores = []
            for provider, result in self.results.items():
                if metric in result.metrics:
                    score = result.metrics[metric]
                    scores.append((provider, score))
                    lines.append(f"  {provider}: {score:.4f}")
            
            if scores:
                # Determine best score
                if "latency" in metric.lower():
                    best_provider = min(scores, key=lambda x: x[1])[0]
                else:
                    best_provider = max(scores, key=lambda x: x[1])[0]
                
                lines.append(f"  Winner: {best_provider}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def get_latency_comparison(self) -> pd.DataFrame:
        """Get latency comparison across models."""
        return self.compare_metrics(["avg_latency", "min_latency", "max_latency"])
    
    def get_quality_comparison(self) -> pd.DataFrame:
        """Get quality metrics comparison."""
        return self.compare_metrics([
            "avg_bleu", "avg_rouge_l", "exact_match"
        ])
    
    def get_token_comparison(self) -> pd.DataFrame:
        """Get token usage comparison."""
        return self.compare_metrics(["avg_tokens", "total_tokens"])
    
    def to_csv(self, output_path: str):
        """Save comparison to CSV file."""
        df = self.compare_metrics()
        df.to_csv(output_path)
    
    def to_json(self, output_path: str):
        """Save comparison to JSON file."""
        import json
        
        comparison = {}
        for provider, result in self.results.items():
            comparison[provider] = result.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
