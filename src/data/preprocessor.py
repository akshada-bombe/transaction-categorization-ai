"""
Data Preprocessing Module
Clean and normalize transaction text
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class TransactionPreprocessor:
    """Preprocess financial transaction text"""
    
    def __init__(self):
        # Common patterns
        self.amount_pattern = r'[\₹\$\€\£]\s*\d+[\d,\.]*'
        self.date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize merchant names
        text = self._normalize_merchants(text)
        
        return text.strip()
    
    def _normalize_merchants(self, text: str) -> str:
        """Normalize common merchant name variations"""
        replacements = {
            r'(?i)swiggy|swigy': 'swiggy',
            r'(?i)zomato|zomto': 'zomato',
            r'(?i)amazon|amzn': 'amazon',
            r'(?i)flipkart|flipkrt': 'flipkart',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def remove_noise(self, text: str) -> str:
        """Remove transaction noise"""
        # Remove amounts
        text = re.sub(self.amount_pattern, '', text)
        
        # Remove dates
        text = re.sub(self.date_pattern, '', text)
        
        # Remove special chars
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess batch of texts"""
        cleaned = []
        for text in texts:
            clean = self.clean_text(text)
            clean = self.remove_noise(clean)
            cleaned.append(clean)
        return cleaned


if __name__ == "__main__":
    # Test
    preprocessor = TransactionPreprocessor()
    test = "SWIGGY BANGALORE ₹450.00"
    print(f"Original: {test}")
    print(f"Cleaned: {preprocessor.clean_text(test)}")