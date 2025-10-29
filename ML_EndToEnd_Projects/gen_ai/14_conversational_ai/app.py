"""
Conversational AI Assistant - End-to-End Implementation
Chatbot with intent recognition, entity extraction, dialogue management, and response generation using transformers.
"""

import os
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Intent recognition and NER models
INTENT_MODEL = 'mrm8488/bert-tiny-finetuned-sms-spam-detection'
NER_MODEL = 'dbmdz/bert-large-cased-finetuned-conll03-english'
DIALOGUE_MODEL = 'facebook/blenderbot-400M-distill'

intent_pipe = pipeline('text-classification', model=INTENT_MODEL, device=0 if torch.cuda.is_available() else -1)
ner_pipe = pipeline('ner', model=NER_MODEL, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
dialogue_pipe = pipeline('conversational', model=DIALOGUE_MODEL, device=0 if torch.cuda.is_available() else -1)

# FastAPI app
app = FastAPI(
    title="Conversational AI API",
    description="Chatbot with intent recognition, entity extraction, dialogue management, and response generation using transformers.",
    version="1.0.0"
)

# Request/response models
class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: Optional[List[Dict]] = None  # [{'role': 'user'/'bot', 'content': str}]

class ChatResponse(BaseModel):
    reply: str
    intent: str
    entities: List[Dict]
    history: List[Dict]

# In-memory dialogue state (for demo)
DIALOGUE_HISTORY = {}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI assistant."""
    try:
        # Intent recognition
        intent_result = intent_pipe(request.message)[0]
        intent = intent_result['label']
        
        # Entity extraction
        entities = ner_pipe(request.message)
        
        # Dialogue history
        user_history = DIALOGUE_HISTORY.get(request.user_id, [])
        if request.history:
            user_history = request.history
        user_history.append({'role': 'user', 'content': request.message})
        
        # Generate response
        from transformers import Conversation
        conversation = Conversation(request.message)
        reply = dialogue_pipe(conversation).generated_responses[-1]
        user_history.append({'role': 'bot', 'content': reply})
        DIALOGUE_HISTORY[request.user_id] = user_history[-10:]  # Keep last 10 turns
        
        return ChatResponse(
            reply=reply,
            intent=intent,
            entities=entities,
            history=user_history
        )
    except Exception as e:
        logger.error(f"Chatbot failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Conversational AI API", "docs": "/docs"}

# Example usage (CLI)
def main():
    print("\n=== Conversational AI Assistant ===")
    user_id = 'demo_user'
    history = []
    while True:
        msg = input("You: ")
        if msg.lower() in ['exit', 'quit']:
            break
        intent_result = intent_pipe(msg)[0]
        intent = intent_result['label']
        entities = ner_pipe(msg)
        from transformers import Conversation
        conversation = Conversation(msg)
        reply = dialogue_pipe(conversation).generated_responses[-1]
        print(f"Bot: {reply}")
        print(f"[Intent: {intent}] [Entities: {entities}]")
        history.append({'role': 'user', 'content': msg})
        history.append({'role': 'bot', 'content': reply})

if __name__ == "__main__":
    main()
