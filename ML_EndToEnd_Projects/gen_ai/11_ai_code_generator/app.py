"""
AI Code Generator - End-to-End Implementation
Automated code generation using transformers, CodeBERT, and GPT models.
"""

import os
import random
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model selection
CODEGEN_MODELS = {
    'codegen-350M': 'Salesforce/codegen-350M-mono',
    'codegen-2B': 'Salesforce/codegen-2B-mono',
    'codebert': 'microsoft/codebert-base',
    'gpt-neo': 'EleutherAI/gpt-neo-1.3B',
    'gpt2': 'gpt2',
    'starcoder': 'bigcode/starcoderbase',
}

# Load default model (small for demo, can be swapped for larger)
def load_codegen_pipeline(model_name: str = 'codegen-350M'):
    model_id = CODEGEN_MODELS.get(model_name, 'Salesforce/codegen-350M-mono')
    logger.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return pipe

# Load once at startup
gen_pipe = load_codegen_pipeline('codegen-350M')

# FastAPI app
app = FastAPI(
    title="AI Code Generator API",
    description="Automated code generation using transformers, CodeBERT, and GPT models.",
    version="1.0.0"
)

# Request/response models
class CodeGenRequest(BaseModel):
    prompt: str
    language: str = 'python'
    max_length: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    model: str = 'codegen-350M'

class CodeGenResponse(BaseModel):
    generated_code: str
    model_used: str
    prompt: str
    language: str

@app.post("/generate", response_model=CodeGenResponse)
async def generate_code(request: CodeGenRequest):
    """Generate code from prompt using selected model."""
    try:
        # Select model
        if request.model not in CODEGEN_MODELS:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not supported.")
        
        # Load/generate pipeline (reuse default for demo)
        pipe = gen_pipe if request.model == 'codegen-350M' else load_codegen_pipeline(request.model)
        
        # Format prompt
        prompt = f"# Language: {request.language}\n# Task: {request.prompt}\n"
        
        # Generate code
        outputs = pipe(
            prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=1,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        code = outputs[0]['generated_text'][len(prompt):].strip()
        
        return CodeGenResponse(
            generated_code=code,
            model_used=request.model,
            prompt=request.prompt,
            language=request.language
        )
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available code generation models."""
    return {"models": list(CODEGEN_MODELS.keys())}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "AI Code Generator API", "docs": "/docs"}

# Example usage (CLI)
def main():
    print("\n=== AI Code Generator ===")
    prompt = input("Enter your code prompt (e.g., 'function to reverse a string'): ")
    language = input("Programming language [python]: ") or 'python'
    model = input(f"Model {list(CODEGEN_MODELS.keys())} [codegen-350M]: ") or 'codegen-350M'
    
    req = CodeGenRequest(prompt=prompt, language=language, model=model)
    
    # Generate code
    outputs = gen_pipe(
        f"# Language: {language}\n# Task: {prompt}\n",
        max_length=req.max_length,
        temperature=req.temperature,
        top_p=req.top_p,
        num_return_sequences=1,
        pad_token_id=gen_pipe.tokenizer.eos_token_id
    )
    code = outputs[0]['generated_text'][len(f"# Language: {language}\n# Task: {prompt}\n"):].strip()
    print("\nGenerated code:\n")
    print(code)

if __name__ == "__main__":
    main()
