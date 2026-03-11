"""Metrics calculation for LLM evaluation."""

import re
import math
from typing import Dict, Any, List, Optional
from collections import Counter


class MetricsCalculator:
    """Calculate various metrics for LLM responses."""
    
    @staticmethod
    def calculate_all(predictions: List[str], references: List[str], 
                     ground_truths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Args:
            predictions: List of model predictions
            references: List of reference texts for comparison
            ground_truths: Optional list of ground truth answers for exact matching
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Basic text metrics
        metrics["avg_response_length"] = MetricsCalculator.avg_response_length(predictions)
        metrics["avg_reference_length"] = MetricsCalculator.avg_response_length(references)
        
        # Similarity metrics
        metrics["avg_bleu"] = MetricsCalculator.avg_bleu_score(predictions, references)
        metrics["avg_rouge_l"] = MetricsCalculator.avg_rouge_l(predictions, references)
        
        # Exact match (if ground truth provided)
        if ground_truths:
            metrics["exact_match"] = MetricsCalculator.exact_match(predictions, ground_truths)
        
        # Quality metrics
        metrics["avg_word_count"] = MetricsCalculator.avg_word_count(predictions)
        metrics["avg_sentence_count"] = MetricsCalculator.avg_sentence_count(predictions)
        
        return metrics
    
    @staticmethod
    def avg_response_length(predictions: List[str]) -> float:
        """Calculate average response length in characters."""
        if not predictions:
            return 0.0
        return sum(len(p) for p in predictions) / len(predictions)
    
    @staticmethod
    def avg_word_count(predictions: List[str]) -> float:
        """Calculate average word count."""
        if not predictions:
            return 0.0
        return sum(len(p.split()) for p in predictions) / len(predictions)
    
    @staticmethod
    def avg_sentence_count(predictions: List[str]) -> float:
        """Calculate average sentence count."""
        if not predictions:
            return 0.0
        sentence_count = sum(len(re.split(r'[.!?]+', p)) for p in predictions)
        return sentence_count / len(predictions)
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple word tokenization."""
        return re.findall(r'\b\w+\b', text.lower())
    
    @staticmethod
    def ngrams(tokens: List[str], n: int) -> Counter:
        """Generate n-grams from tokens."""
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    
    @staticmethod
    def bleu_score(prediction: str, reference: str, n: int = 4) -> float:
        """Calculate BLEU score (simplified version)."""
        pred_tokens = MetricsCalculator.tokenize(prediction)
        ref_tokens = MetricsCalculator.tokenize(reference)
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Calculate precision for each n-gram level
        precisions = []
        for i in range(1, n + 1):
            pred_ngrams = MetricsCalculator.ngrams(pred_tokens, i)
            ref_ngrams = MetricsCalculator.ngrams(ref_tokens, i)
            
            if not pred_ngrams:
                precisions.append(0.0)
                continue
            
            matches = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            precisions.append(matches / total if total > 0 else 0.0)
        
        # Calculate geometric mean of precisions
        precisions = [p for p in precisions if p > 0]
        if not precisions:
            return 0.0
        
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        
        # Calculate brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))
        
        return bp * geo_mean
    
    @staticmethod
    def avg_bleu_score(predictions: List[str], references: List[str]) -> float:
        """Calculate average BLEU score."""
        if not predictions or not references:
            return 0.0
        
        scores = [
            MetricsCalculator.bleu_score(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        return sum(scores) / len(scores)
    
    @staticmethod
    def rouge_l(prediction: str, reference: str) -> float:
        """Calculate ROUGE-L (Longest Common Subsequence)."""
        pred_tokens = MetricsCalculator.tokenize(prediction)
        ref_tokens = MetricsCalculator.tokenize(reference)
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # LCS length
        m, n = len(pred_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # ROUGE-L = LCS / ref_length
        return lcs_length / n if n > 0 else 0.0
    
    @staticmethod
    def avg_rouge_l(predictions: List[str], references: List[str]) -> float:
        """Calculate average ROUGE-L score."""
        if not predictions or not references:
            return 0.0
        
        scores = [
            MetricsCalculator.rouge_l(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        return sum(scores) / len(scores)
    
    @staticmethod
    def exact_match(predictions: List[str], ground_truths: List[str]) -> float:
        """Calculate exact match accuracy."""
        if not predictions or not ground_truths:
            return 0.0
        
        matches = sum(
            1 for pred, truth in zip(predictions, ground_truths)
            if pred.strip().lower() == truth.strip().lower()
        )
        
        return matches / len(predictions)
    
    @staticmethod
    def calculate_latency_stats(responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate latency statistics from responses."""
        latencies = [r.get("latency", 0) for r in responses if r.get("latency")]
        
        if not latencies:
            return {"avg_latency": 0, "min_latency": 0, "max_latency": 0}
        
        return {
            "avg_latency": sum(latencies) / len(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies)
        }
    
    @staticmethod
    def calculate_token_stats(responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate token usage statistics from responses."""
        tokens = [r.get("tokens_used", 0) for r in responses if r.get("tokens_used")]
        
        if not tokens:
            return {"avg_tokens": 0, "total_tokens": 0}
        
        return {
            "avg_tokens": sum(tokens) / len(tokens),
            "total_tokens": sum(tokens)
        }
