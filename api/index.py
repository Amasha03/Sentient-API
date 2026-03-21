from fastapi import FastAPI
from gradio_client import Client
import os

app = FastAPI()

# Map your model IDs to your actual HF Space URLs
MODEL_MAP = {
    "1": ("E-motionAssistant/Space4", None),
    "2": ("E-motionAssistant/Space5", "/predict"),
    "3": ("E-motionAssistant/Space5", "/lambda"),
    "4": ("E-motionAssistant/TherapyEnglish","/lambda"),
    "5": ("E-motionAssistant/TherapyEnglish","/predict"),
    "6": ("E-motionAssistant/Space3","/english_tts"),
    "7": ("E-motionAssistant/Space3","/sinhala_tts"),
    "8": ("E-motionAssistant/Space3","/tamil_tts"),

    
}

@app.get("/api/python/predict")
def predict(model_id: str, message: str):
    model_info = MODEL_MAP.get(model_id)

    if not model_info:
        return {"error": "Invalid Model ID"}
    
    space_path, api_endpoint = model_info
    
    try:
        # Connect to the specific HF Space
        client = Client(space_path)
        result = client.predict(message, api_name=api_endpoint)
        return {"output": result}
    except Exception as e:
        return {"error": str(e)}
