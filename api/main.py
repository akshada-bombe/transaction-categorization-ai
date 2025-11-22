"""
FastAPI Production API - Complete Implementation
Features:
- Transaction categorization
- Explainability (keyword-based)
- Feedback loop
- Taxonomy config support
- CORS enabled
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional
import joblib
import json
import yaml
import os
from datetime import datetime

from src.data.preprocessor import TransactionPreprocessor


# -----------------------------------------------------
# FASTAPI INITIALIZATION
# -----------------------------------------------------
app = FastAPI(
    title="Transaction Categorization API",
    description="AI-powered financial transaction categorization with explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------
# GLOBALS
# -----------------------------------------------------
model = None
preprocessor = None
class_names = []
taxonomy = None
request_count = 0


# -----------------------------------------------------
# REQUEST/RESPONSE MODELS
# -----------------------------------------------------
class TransactionRequest(BaseModel):
    transaction_text: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"transaction_text": "SWIGGY BANGALORE FOOD ORDER"}
        }
    )


class PredictionResponse(BaseModel):
    transaction_text: str
    predicted_category: str
    confidence: float
    explanation: Dict
    timestamp: str


class BatchRequest(BaseModel):
    transactions: List[str]


class FeedbackRequest(BaseModel):
    transaction_text: str
    predicted_category: str
    correct_category: str
    confidence: float
    user_comment: Optional[str] = ""


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    categories_count: int
    total_requests: int
    timestamp: str


# -----------------------------------------------------
# STARTUP: LOAD MODEL + TAXONOMY
# -----------------------------------------------------
@app.on_event("startup")
async def load_model():
    """Load model, taxonomy, and preprocessor on startup"""
    global model, preprocessor, class_names, taxonomy

    try:
        print("\n" + "="*70)
        print("🚀 LOADING API COMPONENTS")
        print("="*70)

        # 1. Load model
        print("\n1. Loading model...")
        data = joblib.load("models/production/baseline_model.pkl")
        model = data["model"]
        model.vectorizer = data["vectorizer"]
        print("   ✓ Model loaded")

        # 2. Load class names
        print("\n2. Loading class names...")
        with open("models/production/class_names.json", "r") as f:
            class_names = json.load(f)["classes"]
        print(f"   ✓ {len(class_names)} categories loaded")

        # 3. Load taxonomy config
        print("\n3. Loading taxonomy...")
        try:
            with open("config/taxonomy.yaml", "r") as f:
                taxonomy = yaml.safe_load(f)
            print(f"   ✓ Taxonomy loaded with {len(taxonomy.get('categories', []))} categories")
        except FileNotFoundError:
            print("   ⚠ taxonomy.yaml not found, using default")
            taxonomy = {"categories": [], "settings": {}}

        # 4. Initialize preprocessor
        print("\n4. Initializing preprocessor...")
        preprocessor = TransactionPreprocessor()
        print("   ✓ Preprocessor ready")

        print("\n" + "="*70)
        print("✅ API READY - Server starting...")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR LOADING MODEL: {e}")
        print("⚠️  Make sure you've trained the model first!")
        print("    Run: python scripts/train_complete.py --data data/raw/transactions_1500.csv\n")


# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------
def generate_explanation(original_text: str, predicted_category: str, confidence: float) -> Dict:
    """Generate keyword-based explanation"""
    
    keywords_found = []
    text_lower = original_text.lower()
    
    # Find matching keywords from taxonomy
    if taxonomy and 'categories' in taxonomy:
        for cat in taxonomy['categories']:
            if cat['name'] == predicted_category:
                for keyword in cat.get('keywords', []):
                    if keyword.lower() in text_lower:
                        keywords_found.append(keyword)
                break
    
    # Determine confidence level
    if confidence >= 0.9:
        confidence_level = "Very High"
    elif confidence >= 0.8:
        confidence_level = "High"
    elif confidence >= 0.7:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    # Generate reason
    if keywords_found:
        reason = f"Classified as '{predicted_category}' based on keywords: {', '.join(keywords_found[:3])}"
    else:
        reason = f"Classified as '{predicted_category}' based on text pattern analysis"
    
    return {
        "method": "keyword_matching",
        "keywords_found": keywords_found[:5],
        "confidence_level": confidence_level,
        "reason": reason
    }


def save_feedback(feedback_data: Dict) -> str:
    """Save feedback to file for continuous learning"""
    
    feedback_dir = "data/feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    
    feedback_file = os.path.join(feedback_dir, "feedback.jsonl")
    
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
    
    return feedback_file


# -----------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Transaction Categorization API",
        "version": "1.0.0",
        "description": "AI-powered financial transaction categorization",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "categories": "/categories",
            "predict": "/predict",
            "batch": "/predict/batch",
            "feedback": "/feedback"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "categories_count": len(class_names),
        "total_requests": request_count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/categories")
def get_categories():
    """Get list of supported categories"""
    
    if not class_names:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    categories_info = []
    
    # Add keyword info from taxonomy
    if taxonomy and 'categories' in taxonomy:
        for cat_name in class_names:
            cat_info = {"name": cat_name, "keywords": []}
            
            for tax_cat in taxonomy['categories']:
                if tax_cat['name'] == cat_name:
                    cat_info['keywords'] = tax_cat.get('keywords', [])[:5]
                    break
            
            categories_info.append(cat_info)
    else:
        categories_info = [{"name": cat, "keywords": []} for cat in class_names]
    
    return {
        "categories": categories_info,
        "total": len(class_names),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(request: TransactionRequest):
    """
    Predict category for single transaction
    
    Returns:
    - predicted_category: The predicted category
    - confidence: Confidence score (0-1)
    - explanation: Why this category was chosen
    """
    global request_count
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.transaction_text.strip():
        raise HTTPException(status_code=400, detail="Transaction text cannot be empty")
    
    try:
        # Increment request counter
        request_count += 1
        
        # Preprocess
        text = preprocessor.clean_text(request.transaction_text)
        text = preprocessor.remove_noise(text)
        
        # Convert to features
        X = model.vectorizer.transform([text])
        
        # Predict
        predicted_category = str(model.predict(X)[0])
        
        # Get confidence
        try:
            proba = model.predict_proba(X)[0]
            confidence = float(max(proba))
        except:
            confidence = 1.0
        
        # Generate explanation
        explanation = generate_explanation(
            request.transaction_text,
            predicted_category,
            confidence
        )
        
        return {
            "transaction_text": request.transaction_text,
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    """
    Predict categories for multiple transactions
    Max 100 transactions per request
    """
    global request_count
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")
    
    if len(request.transactions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 transactions per batch")
    
    try:
        # Increment counter
        request_count += len(request.transactions)
        
        # Preprocess all
        cleaned_texts = []
        for txt in request.transactions:
            t = preprocessor.clean_text(txt)
            t = preprocessor.remove_noise(t)
            cleaned_texts.append(t)
        
        # Convert to features
        X = model.vectorizer.transform(cleaned_texts)
        
        # Predict
        predicted = model.predict(X)
        
        # Get probabilities
        try:
            all_probs = model.predict_proba(X)
        except:
            all_probs = None
        
        # Build response
        predictions = []
        for i, cat in enumerate(predicted):
            
            if all_probs is not None:
                confidence = float(max(all_probs[i]))
            else:
                confidence = 1.0
            
            # Generate explanation
            explanation = generate_explanation(
                request.transactions[i],
                str(cat),
                confidence
            )
            
            predictions.append({
                "transaction_text": request.transactions[i],
                "predicted_category": str(cat),
                "confidence": round(confidence, 4),
                "explanation": explanation
            })
        
        return {
            "predictions": predictions,
            "total": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for model improvement
    
    Used for continuous learning and model retraining
    """
    
    try:
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "transaction_text": request.transaction_text,
            "predicted_category": request.predicted_category,
            "correct_category": request.correct_category,
            "confidence": request.confidence,
            "user_comment": request.user_comment,
            "was_correct": request.predicted_category == request.correct_category
        }
        
        # Save to file
        feedback_file = save_feedback(feedback_data)
        
        # Generate feedback ID
        feedback_id = f"FB_{int(datetime.now().timestamp())}"
        
        return {
            "status": "success",
            "message": "Thank you! Your feedback will help improve the model.",
            "feedback_id": feedback_id,
            "saved_to": feedback_file
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@app.get("/stats")
def get_statistics():
    """Get API usage statistics"""
    
    feedback_count = 0
    
    # Count feedback entries
    try:
        feedback_file = "data/feedback/feedback.jsonl"
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                feedback_count = sum(1 for _ in f)
    except:
        pass
    
    return {
        "total_predictions": request_count,
        "feedback_received": feedback_count,
        "categories_supported": len(class_names),
        "model_status": "loaded" if model else "not loaded",
        "uptime": "active",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/taxonomy")
def get_taxonomy():
    """Get current taxonomy configuration"""
    
    if not taxonomy:
        raise HTTPException(status_code=404, detail="Taxonomy not loaded")
    
    return {
        "taxonomy": taxonomy,
        "categories_count": len(taxonomy.get('categories', [])),
        "timestamp": datetime.now().isoformat()
    }


# -----------------------------------------------------
# ERROR HANDLERS
# -----------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/docs", "/health", "/categories", "/predict"]
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again.",
        "support": "Check logs for details"
    }


# -----------------------------------------------------
# RUN SERVER
# -----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 STARTING TRANSACTION CATEGORIZATION API")
    print("="*70)
    print("\n📍 Server will be available at:")
    print("   • API Docs:  http://localhost:8000/docs")
    print("   • ReDoc:     http://localhost:8000/redoc")
    print("   • Health:    http://localhost:8000/health")
    print("   • Frontend:  Open frontend/index.html in browser")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")