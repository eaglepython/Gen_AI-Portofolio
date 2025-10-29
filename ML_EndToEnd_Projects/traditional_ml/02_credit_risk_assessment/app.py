"""
Credit Risk Assessment - API Service
FastAPI application for serving credit risk predictions.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
import logging
from datetime import datetime
import joblib
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Assessment API",
    description="AI-powered credit risk assessment and loan approval system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
feature_names = []
model_info = {}


# Pydantic models for request/response
class CreditApplication(BaseModel):
    """Credit application data model."""
    # Personal information
    age: int = Field(..., ge=18, le=100, description="Age of applicant")
    income: float = Field(..., gt=0, description="Annual income")
    employment_length: float = Field(..., ge=0, description="Years of employment")
    job_category: str = Field(..., description="Job category")
    
    # Loan information
    credit_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_purpose: str = Field(..., description="Purpose of loan")
    loan_duration: int = Field(..., gt=0, le=120, description="Loan duration in months")
    
    # Credit history
    credit_history_length: float = Field(..., ge=0, description="Years of credit history")
    existing_credits: int = Field(..., ge=0, description="Number of existing credits")
    previous_defaults: int = Field(..., ge=0, description="Number of previous defaults")
    
    # Financial information
    property_type: str = Field(..., description="Property ownership type")
    property_value: float = Field(..., ge=0, description="Property value")
    has_guarantor: bool = Field(..., description="Has guarantor")
    has_co_applicant: bool = Field(..., description="Has co-applicant")
    
    # Banking information
    bank_account_type: str = Field(..., description="Bank account type")
    account_balance: float = Field(..., ge=0, description="Account balance")
    
    @validator('job_category')
    def validate_job_category(cls, v):
        valid_categories = ['Professional', 'Manager', 'Skilled', 'Service', 'Other']
        if v not in valid_categories:
            raise ValueError(f'Job category must be one of: {valid_categories}')
        return v
    
    @validator('loan_purpose')
    def validate_loan_purpose(cls, v):
        valid_purposes = ['Auto', 'Home', 'Personal', 'Business', 'Education']
        if v not in valid_purposes:
            raise ValueError(f'Loan purpose must be one of: {valid_purposes}')
        return v
    
    @validator('property_type')
    def validate_property_type(cls, v):
        valid_types = ['Own', 'Rent', 'Mortgage', 'Other']
        if v not in valid_types:
            raise ValueError(f'Property type must be one of: {valid_types}')
        return v
    
    @validator('bank_account_type')
    def validate_bank_account_type(cls, v):
        valid_types = ['Checking', 'Savings', 'Both', 'None']
        if v not in valid_types:
            raise ValueError(f'Bank account type must be one of: {valid_types}')
        return v


class BatchCreditApplications(BaseModel):
    """Batch credit applications."""
    applications: List[CreditApplication]
    
    @validator('applications')
    def validate_applications(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 applications per batch')
        return v


class CreditRiskResponse(BaseModel):
    """Credit risk assessment response."""
    application_id: str
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0=low risk, 1=high risk)")
    risk_category: str = Field(..., description="Risk category")
    recommendation: str = Field(..., description="Loan recommendation")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    factors: Dict[str, Union[str, float]] = Field(..., description="Risk factors")
    timestamp: str


class BatchCreditRiskResponse(BaseModel):
    """Batch credit risk assessment response."""
    results: List[CreditRiskResponse]
    summary: Dict[str, Union[int, float]]
    timestamp: str


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    model_version: str
    accuracy: float
    auc_score: float
    last_trained: str
    features_count: int


# Utility functions
def preprocess_application(application: CreditApplication) -> pd.DataFrame:
    """Preprocess credit application for model prediction."""
    # Convert to DataFrame
    app_dict = application.dict()
    df = pd.DataFrame([app_dict])
    
    # Calculate derived features (same as in training)
    df['debt_to_income_ratio'] = (df['credit_amount'] * 0.1) / (df['income'] / 12)
    df['income_to_credit_ratio'] = df['income'] / df['credit_amount']
    df['monthly_payment'] = df['credit_amount'] / df['loan_duration']
    df['payment_to_income_ratio'] = (df['monthly_payment'] * 12) / df['income']
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 25, 35, 50, 65, 100], 
                            labels=['Young', 'Young_Adult', 'Middle_Age', 'Senior', 'Elderly'])
    
    # Income groups
    df['income_group'] = pd.cut(df['income'], 
                               bins=[0, 30000, 50000, 75000, 100000, float('inf')], 
                               labels=['Low', 'Lower_Middle', 'Middle', 'Upper_Middle', 'High'])
    
    # Additional engineered features
    df['employment_stability'] = np.where(df['employment_length'] >= 5, 'Stable', 'Unstable')
    df['credit_experience'] = np.where(df['credit_history_length'] >= 3, 'Experienced', 'Novice')
    df['high_risk_job'] = df['job_category'].isin(['Service', 'Other']).astype(int)
    df['multiple_credits'] = (df['existing_credits'] > 2).astype(int)
    df['has_defaults'] = (df['previous_defaults'] > 0).astype(int)
    df['owns_property'] = df['property_type'].isin(['Own', 'Mortgage']).astype(int)
    df['good_banking_relationship'] = df['bank_account_type'].isin(['Checking', 'Both']).astype(int)
    
    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Select only the features used in training
    df_final = df_encoded[feature_names]
    
    return df_final


def interpret_risk_score(risk_score: float) -> tuple:
    """Interpret risk score into category and recommendation."""
    if risk_score <= 0.3:
        return "Low Risk", "Approve"
    elif risk_score <= 0.6:
        return "Medium Risk", "Review"
    else:
        return "High Risk", "Decline"


def get_risk_factors(application: CreditApplication, risk_score: float) -> Dict[str, Union[str, float]]:
    """Identify key risk factors."""
    factors = {}
    
    # Age factor
    if application.age < 25:
        factors['age'] = "Young age increases risk"
    elif application.age > 65:
        factors['age'] = "Advanced age may increase risk"
    
    # Income factor
    if application.income < 30000:
        factors['income'] = "Low income increases risk"
    
    # Employment factor
    if application.employment_length < 2:
        factors['employment'] = "Short employment history increases risk"
    
    # Credit history factor
    if application.credit_history_length < 1:
        factors['credit_history'] = "Limited credit history increases risk"
    
    # Previous defaults
    if application.previous_defaults > 0:
        factors['defaults'] = f"{application.previous_defaults} previous defaults significantly increase risk"
    
    # Debt-to-income ratio
    debt_ratio = (application.credit_amount * 0.1) / (application.income / 12)
    if debt_ratio > 0.4:
        factors['debt_ratio'] = f"High debt-to-income ratio ({debt_ratio:.1%}) increases risk"
    
    # Multiple credits
    if application.existing_credits > 3:
        factors['multiple_credits'] = "Multiple existing credits increase risk"
    
    # Property ownership
    if application.property_type in ['Own', 'Mortgage']:
        factors['property'] = "Property ownership reduces risk"
    
    # Guarantor/co-applicant
    if application.has_guarantor or application.has_co_applicant:
        factors['support'] = "Guarantor/co-applicant reduces risk"
    
    return factors


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Load the trained model and configuration."""
    global model, feature_names, model_info
    
    try:
        logger.info("Loading credit risk model...")
        
        model_dir = Path("models")
        
        # Load the best model
        model_file = model_dir / "best_model.pkl"
        if model_file.exists():
            model = joblib.load(model_file)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found: {model_file}")
            raise FileNotFoundError("Model file not found")
        
        # Load feature names
        feature_file = model_dir / "feature_names.txt"
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(feature_names)} feature names")
        
        # Load model info
        info_file = model_dir / "best_model_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                model_info = json.load(f)
            logger.info("Model info loaded")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources."""
    logger.info("Shutting down Credit Risk Assessment API...")


# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "features_count": len(feature_names)
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if not model_info:
        raise HTTPException(status_code=503, detail="Model information not available")
    
    return ModelInfo(
        model_name=model_info.get('name', 'Unknown'),
        model_version="1.0",
        accuracy=model_info.get('metrics', {}).get('accuracy', 0.0),
        auc_score=model_info.get('metrics', {}).get('auc', 0.0),
        last_trained=model_info.get('timestamp', 'Unknown'),
        features_count=len(feature_names)
    )


@app.post("/predict", response_model=CreditRiskResponse)
async def predict_credit_risk(application: CreditApplication):
    """Predict credit risk for a single application."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate application ID
        app_id = f"APP_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Preprocess application
        app_df = preprocess_application(application)
        
        # Make prediction
        risk_probability = model.predict_proba(app_df)[0][1]  # Probability of default
        
        # Interpret results
        risk_category, recommendation = interpret_risk_score(risk_probability)
        
        # Get confidence (based on prediction probability)
        confidence = max(risk_probability, 1 - risk_probability)
        
        # Get risk factors
        factors = get_risk_factors(application, risk_probability)
        
        return CreditRiskResponse(
            application_id=app_id,
            risk_score=float(risk_probability),
            risk_category=risk_category,
            recommendation=recommendation,
            confidence=float(confidence),
            factors=factors,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchCreditRiskResponse)
async def predict_batch_credit_risk(applications: BatchCreditApplications):
    """Predict credit risk for multiple applications."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        risk_scores = []
        
        for i, application in enumerate(applications.applications):
            # Generate application ID
            app_id = f"BATCH_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i+1:03d}"
            
            # Preprocess and predict
            app_df = preprocess_application(application)
            risk_probability = model.predict_proba(app_df)[0][1]
            
            # Interpret results
            risk_category, recommendation = interpret_risk_score(risk_probability)
            confidence = max(risk_probability, 1 - risk_probability)
            factors = get_risk_factors(application, risk_probability)
            
            results.append(CreditRiskResponse(
                application_id=app_id,
                risk_score=float(risk_probability),
                risk_category=risk_category,
                recommendation=recommendation,
                confidence=float(confidence),
                factors=factors,
                timestamp=datetime.now().isoformat()
            ))
            
            risk_scores.append(risk_probability)
        
        # Calculate summary statistics
        summary = {
            "total_applications": len(applications.applications),
            "avg_risk_score": float(np.mean(risk_scores)),
            "high_risk_count": sum(1 for score in risk_scores if score > 0.6),
            "medium_risk_count": sum(1 for score in risk_scores if 0.3 < score <= 0.6),
            "low_risk_count": sum(1 for score in risk_scores if score <= 0.3),
            "approval_rate": sum(1 for score in risk_scores if score <= 0.6) / len(risk_scores)
        }
        
        return BatchCreditRiskResponse(
            results=results,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """Get API usage statistics."""
    # In a real implementation, this would track actual usage
    return {
        "total_predictions": 0,
        "avg_response_time_ms": 0,
        "model_accuracy": model_info.get('metrics', {}).get('accuracy', 0.0),
        "uptime_hours": 0,
        "last_prediction": None
    }


@app.post("/feedback")
async def submit_feedback(
    application_id: str,
    actual_outcome: bool,
    feedback_notes: Optional[str] = None
):
    """Submit feedback on prediction accuracy."""
    # In a real implementation, this would store feedback for model retraining
    logger.info(f"Feedback received for {application_id}: {actual_outcome}")
    
    return {
        "status": "success",
        "message": "Feedback submitted successfully",
        "application_id": application_id
    }


# Error handlers
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors()
        }
    )


@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()