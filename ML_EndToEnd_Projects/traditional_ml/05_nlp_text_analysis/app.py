"""
NLP Text Analysis System - Complete Implementation
Sentiment analysis, text classification, NER, and summarization.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import spacy
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, pipeline,
    BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
)
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import logging
from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# FastAPI for serving
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from pydantic import BaseModel
import uvicorn

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    logger.warning("Some NLTK downloads failed")


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_process(self, text: str, remove_stopwords: bool = True, 
                           lemmatize: bool = True, stem: bool = False) -> List[str]:
        """Tokenize and process text."""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Stem
        if stem:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def extract_features(self, text: str) -> Dict:
        """Extract various text features."""
        features = {}
        
        # Basic features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return features


class SentimentAnalyzer:
    """Multi-approach sentiment analysis."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.vectorizers = {}
        
        # Load pre-trained transformers
        try:
            self.bert_sentiment = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except:
            logger.warning("Could not load BERT sentiment model")
            self.bert_sentiment = None
        
        # VADER sentiment
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except:
            logger.warning("Could not load VADER sentiment analyzer")
            self.vader = None
    
    def train_classical_model(self, texts: List[str], labels: List[str], 
                            model_type: str = 'logistic') -> Dict:
        """Train classical ML model for sentiment analysis."""
        logger.info(f"Training {model_type} sentiment model")
        
        # Preprocess texts
        processed_texts = [
            ' '.join(self.preprocessor.tokenize_and_process(text))
            for text in texts
        ]
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(processed_texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'naive_bayes':
            model = MultinomialNB()
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store model and vectorizer
        self.models[model_type] = model
        self.vectorizers[model_type] = vectorizer
        
        result = {
            'model_type': model_type,
            'accuracy': accuracy,
            'classification_report': report,
            'feature_count': X.shape[1]
        }
        
        logger.info(f"{model_type} model trained. Accuracy: {accuracy:.4f}")
        return result
    
    def predict_sentiment(self, text: str, method: str = 'ensemble') -> Dict:
        """Predict sentiment using specified method."""
        results = {
            'text': text,
            'predictions': {},
            'ensemble_prediction': None
        }
        
        # VADER sentiment
        if self.vader and method in ['vader', 'ensemble']:
            vader_scores = self.vader.polarity_scores(text)
            results['predictions']['vader'] = {
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'compound': vader_scores['compound'],
                'predicted_class': 'positive' if vader_scores['compound'] > 0.05 else 'negative' if vader_scores['compound'] < -0.05 else 'neutral'
            }
        
        # BERT sentiment
        if self.bert_sentiment and method in ['bert', 'ensemble']:
            try:
                bert_result = self.bert_sentiment(text)[0]
                results['predictions']['bert'] = {
                    pred['label'].lower(): pred['score'] 
                    for pred in bert_result
                }
                results['predictions']['bert']['predicted_class'] = max(
                    results['predictions']['bert'], 
                    key=results['predictions']['bert'].get
                )
            except Exception as e:
                logger.warning(f"BERT prediction failed: {e}")
        
        # Classical models
        for model_name, model in self.models.items():
            if method in [model_name, 'ensemble']:
                try:
                    processed_text = ' '.join(self.preprocessor.tokenize_and_process(text))
                    X = self.vectorizers[model_name].transform([processed_text])
                    
                    probabilities = model.predict_proba(X)[0]
                    classes = model.classes_
                    
                    results['predictions'][model_name] = {
                        classes[i]: float(probabilities[i])
                        for i in range(len(classes))
                    }
                    results['predictions'][model_name]['predicted_class'] = classes[np.argmax(probabilities)]
                    
                except Exception as e:
                    logger.warning(f"{model_name} prediction failed: {e}")
        
        # Ensemble prediction
        if method == 'ensemble' and len(results['predictions']) > 1:
            ensemble_scores = {}
            for pred_method, pred_result in results['predictions'].items():
                for sentiment_class, score in pred_result.items():
                    if sentiment_class != 'predicted_class':
                        if sentiment_class not in ensemble_scores:
                            ensemble_scores[sentiment_class] = []
                        ensemble_scores[sentiment_class].append(score)
            
            # Average scores
            ensemble_avg = {
                sentiment_class: np.mean(scores)
                for sentiment_class, scores in ensemble_scores.items()
            }
            
            results['ensemble_prediction'] = {
                **ensemble_avg,
                'predicted_class': max(ensemble_avg, key=ensemble_avg.get)
            }
        
        return results


class NamedEntityRecognizer:
    """Named Entity Recognition system."""
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Load transformer NER model
        try:
            self.transformer_ner = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
        except:
            logger.warning("Could not load transformer NER model")
            self.transformer_ner = None
    
    def extract_entities_spacy(self, text: str) -> List[Dict]:
        """Extract entities using spaCy."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0  # spaCy doesn't provide confidence scores
            })
        
        return entities
    
    def extract_entities_transformer(self, text: str) -> List[Dict]:
        """Extract entities using transformer model."""
        if not self.transformer_ner:
            return []
        
        try:
            entities = self.transformer_ner(text)
            return [
                {
                    'text': entity['word'],
                    'label': entity['entity_group'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'confidence': entity['score']
                }
                for entity in entities
            ]
        except Exception as e:
            logger.warning(f"Transformer NER failed: {e}")
            return []
    
    def extract_entities_nltk(self, text: str) -> List[Dict]:
        """Extract entities using NLTK."""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            entities = ne_chunk(pos_tags)
            
            extracted_entities = []
            current_entity = []
            current_label = None
            
            for chunk in entities:
                if hasattr(chunk, 'label'):
                    # Named entity
                    current_label = chunk.label()
                    current_entity = [token for token, pos in chunk.leaves()]
                    
                    entity_text = ' '.join(current_entity)
                    extracted_entities.append({
                        'text': entity_text,
                        'label': current_label,
                        'confidence': 1.0
                    })
            
            return extracted_entities
            
        except Exception as e:
            logger.warning(f"NLTK NER failed: {e}")
            return []
    
    def extract_entities(self, text: str, method: str = 'ensemble') -> Dict:
        """Extract entities using specified method."""
        results = {
            'text': text,
            'entities': {}
        }
        
        if method in ['spacy', 'ensemble']:
            results['entities']['spacy'] = self.extract_entities_spacy(text)
        
        if method in ['transformer', 'ensemble']:
            results['entities']['transformer'] = self.extract_entities_transformer(text)
        
        if method in ['nltk', 'ensemble']:
            results['entities']['nltk'] = self.extract_entities_nltk(text)
        
        # Combine entities for ensemble
        if method == 'ensemble':
            all_entities = []
            for method_name, entities in results['entities'].items():
                all_entities.extend(entities)
            
            # Simple deduplication (can be improved)
            unique_entities = []
            seen_texts = set()
            
            for entity in all_entities:
                if entity['text'].lower() not in seen_texts:
                    unique_entities.append(entity)
                    seen_texts.add(entity['text'].lower())
            
            results['ensemble_entities'] = unique_entities
        
        return results


class TextSummarizer:
    """Text summarization system."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        
        # Load transformer summarization model
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
        except:
            logger.warning("Could not load BART summarization model")
            self.summarizer = None
    
    def extractive_summarization(self, text: str, num_sentences: int = 3) -> str:
        """Extractive summarization using TF-IDF."""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Preprocess sentences
        processed_sentences = [
            ' '.join(self.preprocessor.tokenize_and_process(sent))
            for sent in sentences
        ]
        
        # Remove empty sentences
        non_empty_indices = [i for i, sent in enumerate(processed_sentences) if sent.strip()]
        
        if len(non_empty_indices) <= num_sentences:
            return ' '.join([sentences[i] for i in non_empty_indices])
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([processed_sentences[i] for i in non_empty_indices])
        
        # Calculate sentence scores
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_sentence_indices = sorted([non_empty_indices[i] for i in top_indices])
        
        summary = ' '.join([sentences[i] for i in top_sentence_indices])
        return summary
    
    def abstractive_summarization(self, text: str, max_length: int = 130, 
                                min_length: int = 30) -> str:
        """Abstractive summarization using BART."""
        if not self.summarizer:
            return self.extractive_summarization(text)
        
        try:
            # BART has a token limit, so we may need to chunk long texts
            max_input_length = 1024  # Approximate token limit
            
            if len(text.split()) > max_input_length:
                # Split into chunks and summarize each
                words = text.split()
                chunks = [
                    ' '.join(words[i:i+max_input_length])
                    for i in range(0, len(words), max_input_length)
                ]
                
                summaries = []
                for chunk in chunks:
                    chunk_summary = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )[0]['summary_text']
                    summaries.append(chunk_summary)
                
                # If we have multiple summaries, summarize them again
                if len(summaries) > 1:
                    combined_summary = ' '.join(summaries)
                    final_summary = self.summarizer(
                        combined_summary,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )[0]['summary_text']
                    return final_summary
                else:
                    return summaries[0]
            else:
                summary = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
                return summary
                
        except Exception as e:
            logger.warning(f"Abstractive summarization failed: {e}")
            return self.extractive_summarization(text)
    
    def summarize(self, text: str, method: str = 'abstractive', **kwargs) -> Dict:
        """Summarize text using specified method."""
        result = {
            'original_text': text,
            'original_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text))
        }
        
        if method == 'extractive':
            summary = self.extractive_summarization(text, **kwargs)
        elif method == 'abstractive':
            summary = self.abstractive_summarization(text, **kwargs)
        else:
            raise ValueError(f"Unsupported summarization method: {method}")
        
        result.update({
            'summary': summary,
            'summary_length': len(summary),
            'summary_word_count': len(summary.split()),
            'compression_ratio': len(summary) / len(text) if text else 0,
            'method': method
        })
        
        return result


class NLPPipeline:
    """Complete NLP analysis pipeline."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ner = NamedEntityRecognizer()
        self.summarizer = TextSummarizer()
    
    def analyze_text(self, text: str, tasks: List[str] = None) -> Dict:
        """Perform comprehensive text analysis."""
        if tasks is None:
            tasks = ['sentiment', 'entities', 'summary', 'features']
        
        results = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'analysis': {}
        }
        
        # Basic features
        if 'features' in tasks:
            results['analysis']['features'] = self.preprocessor.extract_features(text)
        
        # Sentiment analysis
        if 'sentiment' in tasks:
            results['analysis']['sentiment'] = self.sentiment_analyzer.predict_sentiment(text)
        
        # Named entity recognition
        if 'entities' in tasks:
            results['analysis']['entities'] = self.ner.extract_entities(text)
        
        # Text summarization
        if 'summary' in tasks and len(text.split()) > 20:
            results['analysis']['summary'] = self.summarizer.summarize(text)
        
        return results
    
    def batch_analyze(self, texts: List[str], tasks: List[str] = None) -> List[Dict]:
        """Analyze multiple texts."""
        return [self.analyze_text(text, tasks) for text in texts]


# FastAPI Application
app = FastAPI(
    title="NLP Text Analysis API",
    description="Complete NLP system with sentiment analysis, NER, and summarization",
    version="1.0.0"
)

# Global NLP pipeline
nlp_pipeline = NLPPipeline()

# Request/Response models
class TextAnalysisRequest(BaseModel):
    text: str
    tasks: List[str] = ['sentiment', 'entities', 'summary', 'features']

class SentimentTrainingRequest(BaseModel):
    texts: List[str]
    labels: List[str]
    model_type: str = 'logistic'

class BatchAnalysisRequest(BaseModel):
    texts: List[str]
    tasks: List[str] = ['sentiment', 'entities']

@app.post("/analyze")
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text with specified NLP tasks."""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        results = nlp_pipeline.analyze_text(request.text, request.tasks)
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment/predict")
async def predict_sentiment(text: str = Form(...), method: str = Form(default="ensemble")):
    """Predict sentiment for text."""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        result = nlp_pipeline.sentiment_analyzer.predict_sentiment(text, method)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment/train")
async def train_sentiment_model(request: SentimentTrainingRequest):
    """Train a custom sentiment analysis model."""
    try:
        if len(request.texts) != len(request.labels):
            raise HTTPException(status_code=400, detail="Number of texts and labels must match")
        
        if len(request.texts) < 10:
            raise HTTPException(status_code=400, detail="At least 10 samples required for training")
        
        result = nlp_pipeline.sentiment_analyzer.train_classical_model(
            request.texts, request.labels, request.model_type
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/entities/extract")
async def extract_entities(text: str = Form(...), method: str = Form(default="ensemble")):
    """Extract named entities from text."""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        result = nlp_pipeline.ner.extract_entities(text, method)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_text(
    text: str = Form(...),
    method: str = Form(default="abstractive"),
    max_length: int = Form(default=130),
    min_length: int = Form(default=30),
    num_sentences: int = Form(default=3)
):
    """Summarize text."""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        if len(text.split()) < 10:
            raise HTTPException(status_code=400, detail="Text too short for summarization")
        
        kwargs = {}
        if method == 'abstractive':
            kwargs = {'max_length': max_length, 'min_length': min_length}
        elif method == 'extractive':
            kwargs = {'num_sentences': num_sentences}
        
        result = nlp_pipeline.summarizer.summarize(text, method, **kwargs)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_analyze")
async def batch_analyze(request: BatchAnalysisRequest):
    """Analyze multiple texts."""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(request.texts) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 texts allowed")
        
        results = nlp_pipeline.batch_analyze(request.texts, request.tasks)
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_and_analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    tasks: str = Form(default="sentiment,entities,summary")
):
    """Upload and analyze text file."""
    try:
        # Read file
        content = await file.read()
        text = content.decode('utf-8')
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Parse tasks
        task_list = [task.strip() for task in tasks.split(',')]
        
        # Analyze
        results = nlp_pipeline.analyze_text(text, task_list)
        
        return {
            "filename": file.filename,
            "file_size": len(content),
            "analysis": results
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be text format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/models/status")
async def model_status():
    """Get status of loaded models."""
    status = {
        "sentiment_models": {
            "bert": nlp_pipeline.sentiment_analyzer.bert_sentiment is not None,
            "vader": nlp_pipeline.sentiment_analyzer.vader is not None,
            "classical_models": list(nlp_pipeline.sentiment_analyzer.models.keys())
        },
        "ner_models": {
            "spacy": nlp_pipeline.ner.nlp is not None,
            "transformer": nlp_pipeline.ner.transformer_ner is not None
        },
        "summarization": {
            "bart": nlp_pipeline.summarizer.summarizer is not None
        }
    }
    return status

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "NLP Text Analysis API", "docs": "/docs"}

def main():
    """Main function to run the application."""
    logger.info("Starting NLP Pipeline")
    
    # Example usage
    pipeline = NLPPipeline()
    
    sample_text = """
    Artificial Intelligence is revolutionizing the way we live and work. 
    Companies like Google, Microsoft, and OpenAI are leading the charge in developing 
    advanced AI systems. However, there are concerns about job displacement and 
    ethical implications. The future of AI looks both promising and challenging.
    """
    
    print("\n=== NLP Analysis Example ===")
    results = pipeline.analyze_text(sample_text)
    
    print(f"Text Length: {results['analysis']['features']['word_count']} words")
    
    if 'sentiment' in results['analysis']:
        sentiment = results['analysis']['sentiment']
        if 'ensemble_prediction' in sentiment and sentiment['ensemble_prediction']:
            print(f"Sentiment: {sentiment['ensemble_prediction']['predicted_class']}")
    
    if 'entities' in results['analysis']:
        entities = results['analysis']['entities']
        if 'ensemble_entities' in entities:
            print(f"Entities found: {len(entities['ensemble_entities'])}")
    
    print("\nStarting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()