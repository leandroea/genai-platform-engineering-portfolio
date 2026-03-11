"""Dataset handling for LLM evaluation."""

import json
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path


class Dataset:
    """Handle test datasets for LLM evaluation."""
    
    def __init__(self, data: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize dataset.
        
        Args:
            data: List of dictionaries containing prompt data
        """
        self.data = data or []
    
    @classmethod
    def from_json(cls, file_path: str) -> "Dataset":
        """Load dataset from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data)
    
    @classmethod
    def from_csv(cls, file_path: str) -> "Dataset":
        """Load dataset from CSV file."""
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
        return cls(data)
    
    @classmethod
    def from_list(cls, items: List[Dict[str, Any]]) -> "Dataset":
        """Create dataset from a list of items."""
        return cls(items)
    
    def to_json(self, file_path: str) -> None:
        """Save dataset to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def to_csv(self, file_path: str) -> None:
        """Save dataset to CSV file."""
        df = pd.DataFrame(self.data)
        df.to_csv(file_path, index=False)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame."""
        return pd.DataFrame(self.data)
    
    def get_prompts(self, column: str = "prompt") -> List[str]:
        """Get list of prompts from the dataset."""
        return [item.get(column, "") for item in self.data]
    
    def get_references(self, column: str = "reference") -> List[str]:
        """Get list of reference texts from the dataset."""
        return [item.get(column, "") for item in self.data]
    
    def get_ground_truths(self, column: str = "ground_truth") -> List[str]:
        """Get list of ground truth answers from the dataset."""
        return [item.get(column, "") for item in self.data]
    
    def filter(self, condition: Callable[[Dict[str, Any]], bool]) -> "Dataset":
        """Filter dataset based on a condition."""
        filtered_data = [item for item in self.data if condition(item)]
        return Dataset(filtered_data)
    
    def map(self, func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> "Dataset":
        """Apply a function to each item in the dataset."""
        mapped_data = [func(item) for item in self.data]
        return Dataset(mapped_data)
    
    def shuffle(self, seed: Optional[int] = None) -> "Dataset":
        """Shuffle the dataset."""
        import random
        if seed is not None:
            random.seed(seed)
        shuffled = self.data.copy()
        random.shuffle(shuffled)
        return Dataset(shuffled)
    
    def split(self, ratio: float = 0.8) -> tuple:
        """Split dataset into train and test sets."""
        split_idx = int(len(self.data) * ratio)
        train = Dataset(self.data[:split_idx])
        test = Dataset(self.data[split_idx:])
        return train, test
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item at index."""
        return self.data[index]
    
    def __iter__(self):
        """Iterate over dataset items."""
        return iter(self.data)


def create_sample_dataset() -> Dataset:
    """Create a sample dataset for testing."""
    data = [
        {
            "id": 1,
            "prompt": "What is the capital of France?",
            "reference": "Paris is the capital of France.",
            "ground_truth": "Paris"
        },
        {
            "id": 2,
            "prompt": "Explain photosynthesis in simple terms.",
            "reference": "Photosynthesis is the process by which plants convert sunlight into energy.",
            "ground_truth": "Plants use sunlight to make food"
        },
        {
            "id": 3,
            "prompt": "What year did World War II end?",
            "reference": "World War II ended in 1945.",
            "ground_truth": "1945"
        },
        {
            "id": 4,
            "prompt": "Who wrote Romeo and Juliet?",
            "reference": "William Shakespeare wrote Romeo and Juliet.",
            "ground_truth": "William Shakespeare"
        },
        {
            "id": 5,
            "prompt": "What is the largest planet in our solar system?",
            "reference": "Jupiter is the largest planet in our solar system.",
            "ground_truth": "Jupiter"
        }
    ]
    return Dataset(data)
