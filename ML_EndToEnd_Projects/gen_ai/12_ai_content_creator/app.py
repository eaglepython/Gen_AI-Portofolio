"""
AI Content Creator - End-to-End Implementation
Automated content generation for text and images using GPT, BERT, and diffusion models.
"""

import os
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Text generation models
TEXT_MODELS = {
    'gpt2': 'gpt2',
    'gpt-neo': 'EleutherAI/gpt-neo-1.3B',
    'gpt-j': 'EleutherAI/gpt-j-6B',
    'flan-t5': 'google/flan-t5-base',
    'llama2': 'meta-llama/Llama-2-7b-hf',
}

# Image generation models (diffusers)
IMAGE_MODELS = {
    'stable-diffusion': 'runwayml/stable-diffusion-v1-5',
    'sdxl': 'stabilityai/stable-diffusion-xl-base-1.0',
}

# Load default text pipeline
text_pipe = pipeline('text-generation', model=TEXT_MODELS['gpt2'], tokenizer=TEXT_MODELS['gpt2'], device=0 if torch.cuda.is_available() else -1)

# FastAPI app
app = FastAPI(
    title="AI Content Creator API",
    description="Automated content generation for text and images using GPT, BERT, and diffusion models.",
    version="1.0.0"
)

# Request/response models
class TextGenRequest(BaseModel):
    prompt: str
    max_length: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    model: str = 'gpt2'

class TextGenResponse(BaseModel):
    generated_text: str
    model_used: str
    prompt: str

class ImageGenRequest(BaseModel):
    prompt: str
    num_images: int = 1
    width: int = 512
    height: int = 512
    model: str = 'stable-diffusion'

class ImageGenResponse(BaseModel):
    images: List[str]  # base64-encoded or URLs
    model_used: str
    prompt: str

@app.post("/generate/text", response_model=TextGenResponse)
async def generate_text(request: TextGenRequest):
    """Generate text content from prompt using selected model."""
    try:
        if request.model not in TEXT_MODELS:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not supported.")
        
        # Load/generate pipeline (reuse default for demo)
        pipe = text_pipe if request.model == 'gpt2' else pipeline(
            'text-generation',
            model=TEXT_MODELS[request.model],
            tokenizer=TEXT_MODELS[request.model],
            device=0 if torch.cuda.is_available() else -1
        )
        
        outputs = pipe(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=1,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        text = outputs[0]['generated_text'][len(request.prompt):].strip()
        
        return TextGenResponse(
            generated_text=text,
            model_used=request.model,
            prompt=request.prompt
        )
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/image", response_model=ImageGenResponse)
async def generate_image(request: ImageGenRequest):
    """Generate image(s) from prompt using diffusion models."""
    try:
        from diffusers import StableDiffusionPipeline
        import base64
        import io
        from PIL import Image
        
        if request.model not in IMAGE_MODELS:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not supported.")
        
        model_id = IMAGE_MODELS[request.model]
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        images = []
        for _ in range(request.num_images):
            img = pipe(request.prompt, width=request.width, height=request.height).images[0]
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            images.append(f"data:image/png;base64,{img_b64}")
        
        return ImageGenResponse(
            images=images,
            model_used=request.model,
            prompt=request.prompt
        )
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/text")
async def list_text_models():
    return {"models": list(TEXT_MODELS.keys())}

@app.get("/models/image")
async def list_image_models():
    return {"models": list(IMAGE_MODELS.keys())}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "AI Content Creator API", "docs": "/docs"}

# Example usage (CLI)
def main():
    print("\n=== AI Content Creator ===")
    mode = input("Generate [text/image]?: ").strip().lower()
    if mode == 'text':
        prompt = input("Enter your text prompt: ")
        model = input(f"Model {list(TEXT_MODELS.keys())} [gpt2]: ") or 'gpt2'
        req = TextGenRequest(prompt=prompt, model=model)
        outputs = text_pipe(
            prompt,
            max_length=req.max_length,
            temperature=req.temperature,
            top_p=req.top_p,
            num_return_sequences=1,
            pad_token_id=text_pipe.tokenizer.eos_token_id
        )
        text = outputs[0]['generated_text'][len(prompt):].strip()
        print("\nGenerated text:\n")
        print(text)
    elif mode == 'image':
        prompt = input("Enter your image prompt: ")
        model = input(f"Model {list(IMAGE_MODELS.keys())} [stable-diffusion]: ") or 'stable-diffusion'
        from diffusers import StableDiffusionPipeline
        import base64
        import io
        from PIL import Image
        pipe = StableDiffusionPipeline.from_pretrained(IMAGE_MODELS[model])
        pipe = pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
        img = pipe(prompt).images[0]
        img.show()
        print("Image generated and displayed.")
    else:
        print("Invalid mode.")

if __name__ == "__main__":
    main()
