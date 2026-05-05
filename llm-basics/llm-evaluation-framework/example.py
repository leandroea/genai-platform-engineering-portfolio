"""
Example usage of the LLM Evaluation Framework.

This script demonstrates how to:
1. Load a test dataset
2. Evaluate multiple models
3. Compare results
"""

from src import (
    Config,
    Dataset,
    Evaluator,
    ModelComparator
)


def main():
    """Run the evaluation example."""
    print("=" * 60)
    print("LLM Evaluation Framework - Example Usage")
    print("=" * 60)
    
    # Initialize configuration
    print("\n1. Initializing configuration...")
    config = Config()
    config.validate()
    print(f"   - NVIDIA API configured: {bool(config.nvidia_api_key)}")
    print(f"   - ZAI API configured: {bool(config.zai_api_key)}")
    
    # Load dataset
    print("\n2. Loading test dataset...")
    dataset = Dataset.from_json("data/qa_dataset.json")
    print(f"   - Loaded {len(dataset)} samples")
    
    # Initialize evaluator
    print("\n3. Initializing evaluator...")
    evaluator = Evaluator(config)
    
    # Define models to evaluate
    providers = ["zai"]  # Add "nvidia" if you have NVIDIA API key
    
    # Evaluate models
    print(f"\n4. Evaluating models: {providers}")
    results = evaluator.evaluate_models(providers, dataset, show_progress=True)
    
    # Display results
    print("\n5. Evaluation Results:")
    print("-" * 40)
    
    for provider, result in results.items():
        print(f"\n{provider.upper()}:")
        print(f"   Model: {result.model_name}")
        print(f"   Metrics:")
        for metric, value in result.metrics.items():
            if isinstance(value, float):
                print(f"      {metric}: {value:.4f}")
            else:
                print(f"      {metric}: {value}")
    
    # Compare models (if multiple)
    if len(results) > 1:
        print("\n6. Model Comparison:")
        print("-" * 40)
        comparator = ModelComparator(results)
        print(comparator.generate_report())
    
    # Save results
    print("\n7. Saving results...")
    evaluator.save_results("results/evaluation_results.json")
    print("   - Results saved to results/evaluation_results.json")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
