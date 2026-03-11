# LLM Evaluation Framework

A simple Python framework to evaluate prompts and models automatically. The framework enables sending prompts to multiple LLM providers, collecting responses, calculating metrics, and comparing model performance.

## 1. Problem

Evaluating LLM prompts and models is a critical but challenging task in generative AI development. Key challenges include:

- **Multi-model evaluation**: Need to test the same prompts across different LLM providers (NVIDIA, ZAI, etc.)
- **Automatic metric calculation**: Manually comparing responses is time-consuming and subjective
- **Benchmarking**: Need consistent metrics to compare model performance objectively
- **Scalability**: Must handle multiple prompts and models efficiently

This framework addresses these challenges by providing a unified API to evaluate, measure, and compare LLM outputs automatically.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   LLM Evaluation Framework              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│         ┌──────────────┐    ┌──────────────┐            │
│         │   Config     │    │   Dataset    │            │
│         │  (.env)      │    │  (JSON/CSV)  │            │
│         └──────┬───────┘    └──────┬───────┘            │
│                │                   │                    │
│                v                   v                    │
│         ┌──────────────────────────────────────┐        │
│         │            Evaluator                 │        │
│         │  ┌────────────┐  ┌───────────────┐   │        │
│         │  │ NVIDIA     │  │  ZAI Client   │   │        │
│         │  │ Client     │  │               │   │        │
│         │  └────────────┘  └───────────────┘   │        │
│         └──────────────┬───────────────────────┘        │
│                        │                                │
│                        v                                │
│         ┌──────────────────────────────────┐            │
│         │      Metrics Calculator          │            │
│         │  - BLEU Score                    │            │
│         │  - ROUGE-L                       │            │
│         │  - Exact Match                   │            │
│         │  - Latency Stats                 │            │
│         └──────────────┬───────────────────┘            │
│                        │                                │
│                        v                                │
│         ┌──────────────────────────────────┐            │
│         │     Model Comparator             │            │
│         │  - Ranking                       │            │
│         │  - Winner Detection              │            │
│         │  - Comparison Reports            │            │
│         └──────────────────────────────────┘            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| [`Config`](src/config.py) | Manages API keys and provider settings from `.env` |
| [`ModelClient`](src/models.py) | Abstract base for provider-specific clients |
| [`NVIDIAClient`](src/models.py) | Client for NVIDIA NIM API |
| [`ZAIClient`](src/models.py) | Client for ZAI API |
| [`Dataset`](src/dataset.py) | Handles test data loading and manipulation |
| [`Evaluator`](src/evaluation.py) | Orchestrates model evaluation |
| [`MetricsCalculator`](src/metrics.py) | Computes BLEU, ROUGE-L, Exact Match |
| [`ModelComparator`](src/comparison.py) | Compares results across models |

## 3. Demo

### Quick Start

```python
from src import Config, Dataset, Evaluator, ModelComparator

# 1. Initialize configuration
config = Config()
config.validate()

# 2. Load test dataset
dataset = Dataset.from_json("data/qa_dataset.json")

# 3. Initialize evaluator
evaluator = Evaluator(config)

# 4. Evaluate models
results = evaluator.evaluate_models(["zai"], dataset, show_progress=True)

# 5. Compare results
comparator = ModelComparator(results)
print(comparator.generate_report())
```

### Running Tests

```bash
# Run all tests
.\venv\Scripts\python.exe -m pytest

# Run with coverage
.\venv\Scripts\python.exe -m pytest --cov=src
```

### Example Output

```
============================= test session starts =============================
tests/test_framework.py::TestConfig::test_config_initialization PASSED   [  4%]
tests/test_framework.py::TestConfig::test_config_has_providers PASSED    [  8%]
...
======================== 23 passed in 0.76s ==============================
```

## 4. Key Engineering Challenges

### Challenge 1: Multi-Provider Abstraction
Different LLM providers have different APIs, authentication methods, and response formats. The framework abstracts these differences behind a common `ModelClient` interface while handling provider-specific implementations.

**Solution**: Factory pattern (`get_client()`) creates the appropriate client based on provider name.

### Challenge 2: Metric Calculation
Computing NLP metrics like BLEU and ROUGE-L requires careful tokenization and n-gram matching.

**Solution**: Implemented simplified but effective versions of these metrics that work well for typical LLM evaluation use cases.

### Challenge 3: API Rate Limiting & Errors
LLM APIs can fail due to rate limits, network issues, or invalid responses.

**Solution**: Each client wraps API calls with error handling, returning a `ModelResponse` with error information instead of raising exceptions.

### Challenge 4: Dataset Flexibility
Test datasets come in various formats (JSON, CSV) with different schemas.

**Solution**: `Dataset` class provides uniform access via `get_prompts()`, `get_references()`, and `get_ground_truths()` methods.

## 5. Benchmarks

### Test Results (23 tests passing)

| Category | Tests | Status |
|----------|-------|--------|
| Config | 3 | ✅ PASS |
| Dataset | 7 | ✅ PASS |
| Metrics | 7 | ✅ PASS |
| Evaluation Result | 2 | ✅ PASS |
| Model Comparator | 4 | ✅ PASS |

### Supported Metrics

| Metric | Description | Best For |
|--------|-------------|----------|
| BLEU Score | N-gram precision | Translation quality |
| ROUGE-L | Longest common subsequence | Summarization |
| Exact Match | String equality | Q&A accuracy |
| Latency | Response time | Performance |
| Token Usage | Total tokens | Cost estimation |

## 6. Tradeoffs

### Tradeoff 1: Simplified Metrics vs. Comprehensive Evaluation
- **Decision**: Implemented simplified BLEU/ROUGE-L rather than full implementations
- **Impact**: Good for relative comparisons, but may not match academic benchmarks exactly
- **Mitigation**: Framework is extensible; can add more sophisticated metric implementations

### Tradeoff 2: Sequential vs. Parallel Evaluation
- **Decision**: Evaluations run sequentially to avoid rate limiting
- **Impact**: Slower for large datasets, but more reliable
- **Mitigation**: Can be extended to support async/parallel requests

### Tradeoff 3: Hardcoded Models vs. Dynamic Configuration
- **Decision**: Default models configured in `Config`, but can be extended
- **Impact**: Easy to use out of the box, but requires code changes for new models
- **Mitigation**: `ModelClient` abstraction makes adding new providers straightforward

## 7. Future Work

### Priority 1: Async Support
- Add async/await support for faster parallel model evaluation
- Implement concurrent requests with proper rate limiting

### Priority 2: Additional Metrics
- Add perplexity calculation
- Implement embedding-based similarity metrics
- Add custom scoring functions

### Priority 3: Advanced Dataset Features
- Support for few-shot prompting in datasets
- Dynamic dataset generation from seeds

### Priority 4: Visualization
- Add HTML report generation
- Integration with TensorBoard or MLflow

### Priority 5: CI/CD Integration
- GitHub Actions workflow for automated evaluation
- Performance regression detection
