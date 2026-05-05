"""Tests for the LLM Evaluation Framework."""

import pytest
from src import (
    Config,
    Dataset,
    MetricsCalculator,
    ModelComparator,
    EvaluationResult
)


class TestConfig:
    """Tests for the Config class."""
    
    def test_config_initialization(self):
        """Test config can be initialized."""
        config = Config()
        assert config is not None
    
    def test_config_has_providers(self):
        """Test config has provider configurations."""
        config = Config()
        assert config.nvidia_api_key is not None or config.zai_api_key is not None
    
    def test_get_model_config(self):
        """Test getting model config for a provider."""
        config = Config()
        zai_config = config.get_model_config("zai")
        assert zai_config.provider == "zai"
        assert zai_config.api_key is not None


class TestDataset:
    """Tests for the Dataset class."""
    
    def test_dataset_from_list(self):
        """Test creating dataset from list."""
        data = [
            {"prompt": "Test prompt 1", "reference": "Ref 1"},
            {"prompt": "Test prompt 2", "reference": "Ref 2"}
        ]
        dataset = Dataset.from_list(data)
        assert len(dataset) == 2
    
    def test_dataset_get_prompts(self):
        """Test getting prompts from dataset."""
        data = [
            {"prompt": "Test prompt 1", "reference": "Ref 1"},
            {"prompt": "Test prompt 2", "reference": "Ref 2"}
        ]
        dataset = Dataset.from_list(data)
        prompts = dataset.get_prompts()
        assert prompts == ["Test prompt 1", "Test prompt 2"]
    
    def test_dataset_get_references(self):
        """Test getting references from dataset."""
        data = [
            {"prompt": "Test prompt 1", "reference": "Ref 1"},
            {"prompt": "Test prompt 2", "reference": "Ref 2"}
        ]
        dataset = Dataset.from_list(data)
        references = dataset.get_references()
        assert references == ["Ref 1", "Ref 2"]
    
    def test_dataset_filter(self):
        """Test filtering dataset."""
        data = [
            {"prompt": "Test 1", "category": "a"},
            {"prompt": "Test 2", "category": "b"},
            {"prompt": "Test 3", "category": "a"}
        ]
        dataset = Dataset.from_list(data)
        filtered = dataset.filter(lambda x: x["category"] == "a")
        assert len(filtered) == 2
    
    def test_dataset_to_dataframe(self):
        """Test converting dataset to DataFrame."""
        data = [
            {"prompt": "Test prompt 1", "reference": "Ref 1"},
            {"prompt": "Test prompt 2", "reference": "Ref 2"}
        ]
        dataset = Dataset.from_list(data)
        df = dataset.to_dataframe()
        assert len(df) == 2
        assert "prompt" in df.columns
    
    def test_dataset_shuffle(self):
        """Test shuffling dataset."""
        data = [{"id": i} for i in range(10)]
        dataset = Dataset.from_list(data)
        shuffled = dataset.shuffle(seed=42)
        assert len(shuffled) == 10


class TestMetrics:
    """Tests for the MetricsCalculator class."""
    
    def test_avg_response_length(self):
        """Test average response length calculation."""
        predictions = ["hello", "world"]
        # "hello" = 5 chars, "world" = 5 chars
        length = MetricsCalculator.avg_response_length(predictions)
        assert length == 5.0
    
    def test_avg_word_count(self):
        """Test average word count calculation."""
        predictions = ["hello world", "test response here"]
        count = MetricsCalculator.avg_word_count(predictions)
        assert count == 2.5  # (2 + 3) / 2
    
    def test_tokenize(self):
        """Test tokenization."""
        text = "Hello, world!"
        tokens = MetricsCalculator.tokenize(text)
        assert tokens == ["hello", "world"]
    
    def test_bleu_score_identical(self):
        """Test BLEU score for identical texts."""
        text = "the quick brown fox jumps over the lazy dog"
        score = MetricsCalculator.bleu_score(text, text)
        assert score == 1.0
    
    def test_bleu_score_different(self):
        """Test BLEU score for different texts."""
        pred = "the quick brown fox jumps over the lazy dog"
        ref = "a quick brown animal jumps over a lazy dog"
        score = MetricsCalculator.bleu_score(pred, ref)
        assert 0 < score <= 1
    
    def test_rouge_l_identical(self):
        """Test ROUGE-L for identical texts."""
        text = "the quick brown fox jumps over the lazy dog"
        score = MetricsCalculator.rouge_l(text, text)
        assert score == 1.0
    
    def test_exact_match(self):
        """Test exact match calculation."""
        predictions = ["Paris", "London", "Berlin"]
        ground_truths = ["Paris", "Paris", "Berlin"]
        # Paris matches Paris (1), London doesn't match Paris (0), Berlin matches Berlin (1)
        # 2/3 = 0.666
        em = MetricsCalculator.exact_match(predictions, ground_truths)
        assert em == pytest.approx(2/3)
    
    def test_calculate_all_metrics(self):
        """Test calculating all metrics at once."""
        predictions = ["Paris is the capital of France"]
        references = ["Paris is the capital city of France"]
        ground_truths = ["Paris"]
        
        metrics = MetricsCalculator.calculate_all(
            predictions, references, ground_truths
        )
        
        assert "avg_bleu" in metrics
        assert "avg_rouge_l" in metrics
        assert "exact_match" in metrics


class TestEvaluationResult:
    """Tests for the EvaluationResult class."""
    
    def test_evaluation_result_creation(self):
        """Test creating evaluation result."""
        result = EvaluationResult("test-model", "nvidia")
        assert result.model_name == "test-model"
        assert result.provider == "nvidia"
        assert len(result.responses) == 0
    
    def test_evaluation_result_to_dict(self):
        """Test converting result to dictionary."""
        result = EvaluationResult("test-model", "nvidia")
        result_dict = result.to_dict()
        assert "model_name" in result_dict
        assert "provider" in result_dict
        assert "metrics" in result_dict


class TestModelComparator:
    """Tests for the ModelComparator class."""
    
    def test_comparator_creation(self):
        """Test creating comparator."""
        result1 = EvaluationResult("model1", "nvidia")
        result2 = EvaluationResult("model2", "zai")
        
        results = {
            "nvidia": result1,
            "zai": result2
        }
        
        comparator = ModelComparator(results)
        assert comparator.results == results
    
    def test_get_winner(self):
        """Test getting winner for a metric."""
        result1 = EvaluationResult("model1", "nvidia")
        result1.metrics = {"accuracy": 0.8}
        
        result2 = EvaluationResult("model2", "zai")
        result2.metrics = {"accuracy": 0.9}
        
        results = {"nvidia": result1, "zai": result2}
        comparator = ModelComparator(results)
        
        winner = comparator.get_winner("accuracy")
        assert winner == "zai"
    
    def test_rank_models(self):
        """Test ranking models."""
        result1 = EvaluationResult("model1", "nvidia")
        result1.metrics = {"accuracy": 0.8}
        
        result2 = EvaluationResult("model2", "zai")
        result2.metrics = {"accuracy": 0.9}
        
        result3 = EvaluationResult("model3", "other")
        result3.metrics = {"accuracy": 0.7}
        
        results = {
            "nvidia": result1,
            "zai": result2,
            "other": result3
        }
        comparator = ModelComparator(results)
        
        ranking = comparator.rank_models("accuracy")
        assert ranking[0] == ("zai", 0.9)
        assert ranking[1] == ("nvidia", 0.8)
        assert ranking[2] == ("other", 0.7)
    
    def test_latency_winner(self):
        """Test latency winner (lower is better)."""
        result1 = EvaluationResult("model1", "nvidia")
        result1.metrics = {"avg_latency": 1.5}
        
        result2 = EvaluationResult("model2", "zai")
        result2.metrics = {"avg_latency": 1.0}
        
        results = {"nvidia": result1, "zai": result2}
        comparator = ModelComparator(results)
        
        winner = comparator.get_winner("avg_latency")
        assert winner == "zai"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
