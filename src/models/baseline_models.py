
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import List, Dict
import joblib


class BaselineClassifier:
    """Baseline ML classifier using TF-IDF + Logistic Regression"""
    
    def __init__(self, model_type: str = 'logistic'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2
        )
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
    
    def train(self, texts: List[str], labels: List[str]):
        """Train the model"""
        print(f"🚀 Training {self.model_type} model...")
        
        # Vectorize
        X = self.vectorizer.fit_transform(texts)
        
        # Train
        self.model.fit(X, labels)
        
        print(f"✓ Training complete!")
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """Predict categories"""
        # Vectorize
        X = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'predicted_category': predictions[i],
                'confidence': float(probabilities[i].max()),
                'all_probabilities': probabilities[i].tolist()
            })
        
        return results
    
    def save(self, path: str):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        data = joblib.load(path)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        print(f"✓ Model loaded from {path}")


if __name__ == "__main__":
    # Test
    texts = ["swiggy bangalore", "amazon prime", "hp petrol pump"]
    labels = ["Food & Drinks", "Shopping", "Fuel"]
    
    clf = BaselineClassifier()
    clf.train(texts, labels)
    
    # Predict
    test_texts = ["swiggy order"]
    results = clf.predict(test_texts)
    print(f"\nPrediction: {results[0]['predicted_category']}")
    print(f"Confidence: {results[0]['confidence']:.2%}")