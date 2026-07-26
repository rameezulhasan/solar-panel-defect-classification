from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io

# ============================================
# App initialize karo
# ============================================
app = FastAPI(title="Solar Panel Defect Classification API")

# ============================================
# Model aur class names load karo (app start hote hi ek hi baar load hoga)
# ============================================
model = tf.keras.models.load_model("best_model_efficientnetb0_finetuned.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ============================================
# Preprocessing function - same jo Streamlit mein tha
# ============================================
def preprocess_image(image: Image.Image):
    
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    
    # Batch dimension add karo - model batch expect karta hai
    image = np.expand_dims(image, axis=0)
    
    return image

# ============================================
# Health check endpoint - Docker/monitoring ke liye useful
# ============================================
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Solar Panel Defect Classification API is running"}

# ============================================
# Prediction endpoint
# ============================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # Uploaded file ko read karo
    contents = await file.read()
    
    # Bytes se PIL Image banao
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess karo
    processed_image = preprocess_image(image)
    
    # Prediction lo
    prediction = model.predict(processed_image, verbose=0)
    
    # Highest probability wali class nikalo
    predicted_index = int(np.argmax(prediction[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index])
    
    # Sab classes ki probability bhi bhej do (JSON format mein)
    all_probabilities = {}
    for i in range(len(class_names)):
        all_probabilities[class_names[i]] = float(prediction[0][i])
    
    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probabilities
    })