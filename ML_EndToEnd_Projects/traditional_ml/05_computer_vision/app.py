"""
Computer Vision System - Complete Implementation
Object detection, image classification, and face recognition system.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import face_recognition
import dlib
from pathlib import Path
import pickle
import logging
from datetime import datetime
import json
import os
from typing import List, Dict, Tuple, Optional
import base64
from io import BytesIO
from PIL import Image

# FastAPI for serving
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Object detection with YOLO
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectDetectionEngine:
    """YOLO-based object detection system."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.classes = []
        self.load_model(model_path)
    
    def load_model(self, model_path: str = None):
        """Load YOLO model."""
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                # Load pre-trained YOLOv8 model
                self.model = YOLO('yolov8n.pt')  # nano version for speed
            
            logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            # Fallback to OpenCV DNN
            self.load_opencv_dnn()
    
    def load_opencv_dnn(self):
        """Load OpenCV DNN model as fallback."""
        try:
            # Download YOLO weights and config if not present
            weights_path = "yolo/yolov3.weights"
            config_path = "yolo/yolov3.cfg"
            classes_path = "yolo/coco.names"
            
            if os.path.exists(weights_path) and os.path.exists(config_path):
                self.net = cv2.dnn.readNet(weights_path, config_path)
                
                with open(classes_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
                
                logger.info("OpenCV DNN model loaded as fallback")
            else:
                logger.warning("No object detection model available")
                
        except Exception as e:
            logger.error(f"Error loading OpenCV DNN: {e}")
    
    def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """Detect objects in image."""
        try:
            if hasattr(self, 'model') and self.model:
                # Use YOLO model
                results = self.model(image)
                detections = []
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            confidence = float(box.conf[0])
                            if confidence >= confidence_threshold:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                class_id = int(box.cls[0])
                                class_name = self.model.names[class_id]
                                
                                detections.append({
                                    'class': class_name,
                                    'confidence': confidence,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                                })
                
                return detections
            
            elif hasattr(self, 'net'):
                # Use OpenCV DNN as fallback
                return self.detect_with_opencv_dnn(image, confidence_threshold)
            
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []
    
    def detect_with_opencv_dnn(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """Object detection using OpenCV DNN."""
        height, width = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run forward pass
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                
                detections.append({
                    'class': class_name,
                    'confidence': confidences[i],
                    'bbox': [x, y, x + w, y + h]
                })
        
        return detections


class ImageClassificationEngine:
    """Transfer learning based image classification."""
    
    def __init__(self):
        self.model = None
        self.classes = []
        self.input_shape = (224, 224, 3)
        
    def create_model(self, num_classes: int, base_model_name: str = 'resnet50'):
        """Create classification model using transfer learning."""
        logger.info(f"Creating model with {base_model_name} backbone")
        
        # Load pre-trained base model
        if base_model_name.lower() == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name.lower() == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name.lower() == 'mobilenetv2':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"Model created with {num_classes} classes")
        return model
    
    def train_model(self, train_dir: str, val_dir: str, epochs: int = 10, batch_size: int = 32):
        """Train the classification model."""
        logger.info("Starting model training")
        
        # Data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        self.classes = list(train_generator.class_indices.keys())
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            verbose=1
        )
        
        logger.info("Model training completed")
        return history
    
    def predict_image(self, image: np.ndarray) -> Dict:
        """Classify a single image."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.resize(image, self.input_shape[:2])
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        result = {
            'predicted_class': self.classes[class_idx] if self.classes else f"class_{class_idx}",
            'confidence': confidence,
            'all_predictions': {
                self.classes[i] if self.classes else f"class_{i}": float(predictions[0][i])
                for i in range(len(predictions[0]))
            }
        }
        
        return result
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model:
            self.model.save(filepath)
            
            # Save class names
            class_file = filepath.replace('.h5', '_classes.json')
            with open(class_file, 'w') as f:
                json.dump(self.classes, f)
                
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            
            # Load class names
            class_file = filepath.replace('.h5', '_classes.json')
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    self.classes = json.load(f)
            
            logger.info(f"Model loaded from {filepath}")


class FaceRecognitionEngine:
    """Face detection and recognition system."""
    
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_detector = dlib.get_frontal_face_detector()
        
    def add_known_face(self, image_path: str, name: str):
        """Add a known face to the database."""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                logger.info(f"Added face for {name}")
                return True
            else:
                logger.warning(f"No face found in {image_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding face {name}: {e}")
            return False
    
    def recognize_faces(self, image: np.ndarray) -> List[Dict]:
        """Recognize faces in image."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            results = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                name = "Unknown"
                confidence = 0.0
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1.0 - face_distances[best_match_index]
                
                results.append({
                    'name': name,
                    'confidence': float(confidence),
                    'bbox': [left, top, right, bottom]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return []
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces without recognition."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            
            results = []
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                results.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': 1.0  # Dlib doesn't provide confidence scores
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def save_encodings(self, filepath: str):
        """Save face encodings database."""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Face encodings saved to {filepath}")
    
    def load_encodings(self, filepath: str):
        """Load face encodings database."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.known_face_encodings = data['encodings']
            self.known_face_names = data['names']
            
            logger.info(f"Loaded {len(self.known_face_names)} face encodings")


class ComputerVisionPipeline:
    """Complete computer vision pipeline."""
    
    def __init__(self):
        self.object_detector = ObjectDetectionEngine()
        self.image_classifier = ImageClassificationEngine()
        self.face_recognizer = FaceRecognitionEngine()
        
    def process_image(self, image: np.ndarray, tasks: List[str] = None) -> Dict:
        """Process image with specified tasks."""
        if tasks is None:
            tasks = ['object_detection', 'face_detection']
        
        results = {
            'image_shape': image.shape,
            'timestamp': datetime.now().isoformat()
        }
        
        # Object detection
        if 'object_detection' in tasks:
            objects = self.object_detector.detect_objects(image)
            results['objects'] = objects
        
        # Face detection/recognition
        if 'face_detection' in tasks:
            faces = self.face_recognizer.detect_faces(image)
            results['faces'] = faces
        
        if 'face_recognition' in tasks:
            faces = self.face_recognizer.recognize_faces(image)
            results['recognized_faces'] = faces
        
        # Image classification
        if 'classification' in tasks and self.image_classifier.model:
            classification = self.image_classifier.predict_image(image)
            results['classification'] = classification
        
        return results
    
    def draw_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results on image."""
        output_image = image.copy()
        
        # Draw object detections
        if 'objects' in results:
            for obj in results['objects']:
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{obj['class']}: {obj['confidence']:.2f}"
                cv2.putText(output_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw face detections
        if 'faces' in results:
            for face in results['faces']:
                x1, y1, x2, y2 = face['bbox']
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(output_image, "Face", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw recognized faces
        if 'recognized_faces' in results:
            for face in results['recognized_faces']:
                x1, y1, x2, y2 = face['bbox']
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{face['name']}: {face['confidence']:.2f}"
                cv2.putText(output_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return output_image


# FastAPI Application
app = FastAPI(
    title="Computer Vision API",
    description="Complete computer vision system with object detection, classification, and face recognition",
    version="1.0.0"
)

# Global CV pipeline
cv_pipeline = ComputerVisionPipeline()

# Request/Response models
class ImageProcessRequest(BaseModel):
    tasks: List[str] = ['object_detection', 'face_detection']

class DetectionResult(BaseModel):
    objects: Optional[List[Dict]] = None
    faces: Optional[List[Dict]] = None
    recognized_faces: Optional[List[Dict]] = None
    classification: Optional[Dict] = None
    image_shape: List[int]
    processing_time: float

def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image data."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return cv_image
        
    except Exception as e:
        raise ValueError(f"Error decoding image: {e}")

def encode_image(image: np.ndarray) -> str:
    """Encode image to base64."""
    try:
        # Convert to PIL Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{encoded_image}"
        
    except Exception as e:
        raise ValueError(f"Error encoding image: {e}")

@app.post("/detect", response_model=DetectionResult)
async def detect_objects_and_faces(
    file: UploadFile = File(...),
    tasks: str = Form(default="object_detection,face_detection")
):
    """Process uploaded image for object detection and face recognition."""
    try:
        start_time = datetime.now()
        
        # Read and decode image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Parse tasks
        task_list = [task.strip() for task in tasks.split(',')]
        
        # Process image
        results = cv_pipeline.process_image(image, task_list)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DetectionResult(
            objects=results.get('objects'),
            faces=results.get('faces'),
            recognized_faces=results.get('recognized_faces'),
            classification=results.get('classification'),
            image_shape=list(results['image_shape']),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_with_visualization")
async def detect_with_visualization(
    file: UploadFile = File(...),
    tasks: str = Form(default="object_detection,face_detection")
):
    """Process image and return results with visualization."""
    try:
        # Read and decode image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Parse tasks
        task_list = [task.strip() for task in tasks.split(',')]
        
        # Process image
        results = cv_pipeline.process_image(image, task_list)
        
        # Draw results on image
        output_image = cv_pipeline.draw_results(image, results)
        
        # Encode output image
        encoded_image = encode_image(output_image)
        
        return JSONResponse({
            "results": results,
            "annotated_image": encoded_image
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_face")
async def add_face(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    """Add a known face to the recognition database."""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{name}_{datetime.now().timestamp()}.jpg"
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Add face to database
        success = cv_pipeline.face_recognizer.add_known_face(temp_path, name)
        
        # Clean up
        os.remove(temp_path)
        
        if success:
            return {"message": f"Face added for {name}", "success": True}
        else:
            return {"message": f"No face found in image for {name}", "success": False}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Computer Vision API", "docs": "/docs"}

def main():
    """Main function to run the application."""
    logger.info("Starting Computer Vision Pipeline")
    
    # Example usage
    pipeline = ComputerVisionPipeline()
    
    print("\n=== Computer Vision System Ready ===")
    print("Capabilities:")
    print("- Object Detection (YOLO)")
    print("- Face Detection & Recognition")
    print("- Image Classification (Transfer Learning)")
    print("- Real-time Processing")
    
    print("\nStarting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()