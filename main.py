from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# Mount static folder for serving HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model once at startup
model_path = os.path.join("model", "fruit_ripeness_model2.h5")
model = load_model(model_path)
class_indices = {0: 'fresh', 1: 'rotten'}

# Serve the frontend page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Image prediction endpoint
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((64, 64))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_indices[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return JSONResponse(content={
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%"
    })
