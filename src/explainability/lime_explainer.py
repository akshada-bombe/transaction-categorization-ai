import numpy as np
from lime.lime_text import LimeTextExplainer
from typing import Dict, List, Callable
import pandas as pd


class TransactionExplainer:
    """
    LIME-based explainability for transaction categorization
    Helps users understand WHY a transaction was categorized
    """
    
    def __init__(self, class_names: List[str], predict_fn: Callable):
        """
        Initialize explainer
        
        Args:
            class_names: List of category names
            predict_fn: Function that takes text and returns probabilities
        """
        self.class_names = class_names
        self.predict_fn = predict_fn
        
        # Initialize LIME explainer
        self.explainer = LimeTextExplainer(
            class_names=class_names,
            bow=False,  # Use more sophisticated text representation
            random_state=42
        )
    
    def explain_prediction(self, 
                          text: str,
                          num_features: int = 10,
                          num_samples: int = 1000) -> Dict:
        """
        Generate explanation for a single prediction
        
        Args:
            text: Transaction text to explain
            num_features: Number of words to highlight in explanation
            num_samples: Number of perturbed samples for LIME
        
        Returns:
            Dict containing explanation details
        """
        
        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            text,
            self.predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Get predicted class
        predicted_class = exp.available_labels()[0]
        predicted_class_name = self.class_names[predicted_class]
        
        # Extract feature importances
        feature_weights = exp.as_list(label=predicted_class)
        
        # Separate positive and negative contributions
        positive_features = [(f, w) for f, w in feature_weights if w > 0]
        negative_features = [(f, w) for f, w in feature_weights if w < 0]
        
        # Sort by absolute importance
        positive_features.sort(key=lambda x: abs(x[1]), reverse=True)
        negative_features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Build explanation
        explanation = {
            'text': text,
            'predicted_category': predicted_class_name,
            'explanation_type': 'LIME',
            'key_words_supporting': [
                {'word': f, 'importance': round(w, 4)} 
                for f, w in positive_features[:5]
            ],
            'key_words_against': [
                {'word': f, 'importance': round(w, 4)} 
                for f, w in negative_features[:5]
            ],
            'interpretation': self._generate_interpretation(
                text, predicted_class_name, positive_features, negative_features
            )
        }
        
        return explanation
    
    def _generate_interpretation(self, 
                                text: str,
                                category: str,
                                positive: List,
                                negative: List) -> str:
        """Generate human-readable interpretation"""
        
        if not positive:
            return f"Classified as '{category}' based on overall text pattern."
        
        top_words = [f"'{f}'" for f, _ in positive[:3]]
        
        if len(top_words) == 1:
            words_str = top_words[0]
        elif len(top_words) == 2:
            words_str = f"{top_words[0]} and {top_words[1]}"
        else:
            words_str = f"{', '.join(top_words[:-1])}, and {top_words[-1]}"
        
        interpretation = (
            f"Classified as '{category}' primarily because the text contains {words_str}, "
            f"which are strong indicators of this category."
        )
        
        if negative:
            neg_word = negative[0][0]
            interpretation += (
                f" However, the presence of '{neg_word}' slightly reduces the confidence."
            )
        
        return interpretation
    
    def explain_batch(self, texts: List[str], num_features: int = 10) -> pd.DataFrame:
        """Generate explanations for multiple transactions"""
        
        explanations = []
        
        for text in texts:
            try:
                exp = self.explain_prediction(text, num_features=num_features)
                explanations.append({
                    'text': text,
                    'predicted_category': exp['predicted_category'],
                    'top_feature': exp['key_words_supporting'][0]['word'] if exp['key_words_supporting'] else 'N/A',
                    'interpretation': exp['interpretation']
                })
            except Exception as e:
                explanations.append({
                    'text': text,
                    'predicted_category': 'ERROR',
                    'top_feature': 'N/A',
                    'interpretation': f'Error: {str(e)}'
                })
        
        return pd.DataFrame(explanations)
    
    def visualize_explanation(self, text: str, save_path: str = None):
        """
        Generate HTML visualization of explanation
        
        Args:
            text: Transaction text
            save_path: Optional path to save HTML
        """
        
        exp = self.explainer.explain_instance(
            text,
            self.predict_fn,
            num_features=10
        )
        
        # Generate HTML
        html = exp.as_html()
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html)
            print(f"✓ Visualization saved to {save_path}")
        
        return html


class KeywordAttributor:
    """
    Simple keyword-based attribution for fast explanation
    Complements LIME for real-time inference
    """
    
    def __init__(self, category_keywords: Dict[str, List[str]]):
        """
        Initialize with category keyword mappings
        
        Args:
            category_keywords: Dict mapping categories to their keywords
        """
        self.category_keywords = category_keywords
    
    def explain(self, text: str, predicted_category: str) -> Dict:
        """Generate keyword-based explanation"""
        
        text_lower = text.lower()
        
        # Find matching keywords for predicted category
        category_keywords = self.category_keywords.get(predicted_category, [])
        matched_keywords = [kw for kw in category_keywords if kw.lower() in text_lower]
        
        # Calculate keyword coverage
        coverage = len(matched_keywords) / len(category_keywords) if category_keywords else 0
        
        explanation = {
            'method': 'keyword_attribution',
            'matched_keywords': matched_keywords[:5],
            'keyword_coverage': round(coverage, 2),
            'interpretation': self._generate_keyword_interpretation(
                predicted_category, matched_keywords
            )
        }
        
        return explanation
    
    def _generate_keyword_interpretation(self, category: str, keywords: List[str]) -> str:
        """Generate interpretation from keywords"""
        
        if not keywords:
            return f"No specific keywords found, but text pattern matches '{category}'."
        
        if len(keywords) == 1:
            return f"Classified as '{category}' because it contains '{keywords[0]}'."
        
        keyword_str = ', '.join(f"'{kw}'" for kw in keywords[:3])
        return f"Classified as '{category}' based on keywords: {keyword_str}."


class ExplanationComparator:
    """
    Compare explanations across multiple models
    Useful for ensemble models
    """
    
    @staticmethod
    def compare_explanations(explanations: List[Dict]) -> Dict:
        """Compare multiple explanations and find consensus"""
        
        # Extract all mentioned features
        all_features = set()
        for exp in explanations:
            for feat in exp.get('key_words_supporting', []):
                all_features.add(feat['word'])
        
        # Count feature frequency
        feature_counts = {feat: 0 for feat in all_features}
        
        for exp in explanations:
            for feat in exp.get('key_words_supporting', []):
                feature_counts[feat['word']] += 1
        
        # Find consensus features (appear in majority of explanations)
        threshold = len(explanations) / 2
        consensus_features = [
            feat for feat, count in feature_counts.items() 
            if count >= threshold
        ]
        
        return {
            'consensus_features': consensus_features,
            'total_features': len(all_features),
            'consensus_ratio': len(consensus_features) / len(all_features) if all_features else 0
        }


if __name__ == "__main__":
    # Example usage
    
    # Mock predict function
    def mock_predict(texts):
        """Mock prediction function for testing"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Return random probabilities for 3 classes
        return np.random.dirichlet(np.ones(3), size=len(texts))
    
    # Initialize explainer
    class_names = ['Food & Drinks', 'Shopping', 'Fuel']
    explainer = TransactionExplainer(class_names, mock_predict)
    
    # Test explanation
    sample_text = "SWIGGY BANGALORE FOOD ORDER"
    explanation = explainer.explain_prediction(sample_text, num_features=5)
    
    print("="*70)
    print("EXPLAINABILITY DEMO")
    print("="*70)
    print(f"Text: {explanation['text']}")
    print(f"Predicted: {explanation['predicted_category']}")
    print(f"\nInterpretation:")
    print(explanation['interpretation'])
    print("\n✓ Explainability module ready")