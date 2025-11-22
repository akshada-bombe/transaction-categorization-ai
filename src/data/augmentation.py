"""
Data Augmentation Module
Expand dataset from 1500 to 4500 samples
"""

import pandas as pd
import random
from typing import List


class TransactionAugmenter:
    """Augment transaction data"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        # Templates for augmentation
        self.templates = {
            'food_drinks': [
                "{merchant} payment",
                "payment to {merchant}",
                "{merchant} order",
                "upi {merchant}",
            ],
            'shopping': [
                "{merchant} purchase",
                "payment to {merchant}",
                "{merchant} order",
            ],
            'groceries': [
                "{merchant} grocery",
                "{merchant} shopping",
            ],
            'fuel': [
                "{merchant} petrol pump",
                "{merchant} fuel",
            ],
            'travel': [
                "{merchant} ride",
                "{merchant} booking",
            ],
            'bills_recharge': [
                "{merchant} recharge",
                "{merchant} bill payment",
            ],
            'entertainment': [
                "{merchant} subscription",
                "{merchant} payment",
            ],
            'medical': [
                "{merchant} pharmacy",
                "{merchant} medical",
            ],
            'online_services': [
                "{merchant} subscription",
                "{merchant} service",
            ],
            'banking_charges': [
                "{merchant} charges",
                "{merchant} fee",
            ],
        }
    
    def extract_merchant(self, text: str) -> str:
        """Extract merchant name"""
        words = text.lower().split()
        skip = {'payment', 'to', 'upi', 'debit', 'credit'}
        merchant = [w for w in words[:2] if w not in skip]
        return ' '.join(merchant) if merchant else words[0] if words else 'merchant'
    
    def augment_dataset(self, df: pd.DataFrame, target_multiplier: int = 3) -> pd.DataFrame:
        """Augment dataset"""
        augmented_data = []
        
        print(f"🔄 Augmenting dataset {target_multiplier}x...")
        
        # Keep original
        for _, row in df.iterrows():
            augmented_data.append({
                'transaction_text': row['transaction_text'],
                'category': row['category'],
                'is_augmented': False
            })
        
        # Generate augmented
        for _, row in df.iterrows():
            text = row['transaction_text']
            category = row['category']
            merchant = self.extract_merchant(text)
            
            # Get category key
            cat_key = category.lower().replace(' ', '_').replace('&', '').replace('  ', '_')
            templates = self.templates.get(cat_key, ["{merchant}"])
            
            # Generate variations
            for template in templates[:2]:  # 2 variations per sample
                variation = template.format(merchant=merchant)
                augmented_data.append({
                    'transaction_text': variation,
                    'category': category,
                    'is_augmented': True
                })
                
                if len(augmented_data) >= len(df) * target_multiplier:
                    break
            
            if len(augmented_data) >= len(df) * target_multiplier:
                break
        
        # Create dataframe
        result_df = pd.DataFrame(augmented_data[:len(df) * target_multiplier])
        
        print(f"✓ Augmentation complete: {len(df)} → {len(result_df)} samples")
        
        return result_df


if __name__ == "__main__":
    # Test
    df = pd.DataFrame({
        'transaction_text': ['SWIGGY BANGALORE', 'AMAZON PRIME'],
        'category': ['Food & Drinks', 'Shopping']
    })
    
    augmenter = TransactionAugmenter()
    augmented = augmenter.augment_dataset(df, target_multiplier=3)
    print(f"\nAugmented samples: {len(augmented)}")
    print(augmented.head())