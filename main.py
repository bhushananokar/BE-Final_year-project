from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from typing import Dict, List
import json
import os
from fastapi.responses import PlainTextResponse 
genai.configure(api_key='AIzaSyDdLLzxrYEXWFOZYTc1kOuMoQVNsdV6qNo')
model = genai.GenerativeModel('gemini-pro')
app = FastAPI(title="Disease Prediction API",
             description="API for predicting diseases based on symptoms using Google's Gemini model")
class SymptomsInput(BaseModel):
    symptoms: str
class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    remedies: List[str]
    precautions: List[str]
def generate_prompt(symptoms: str) -> str:
    return f"""Based on the following symptoms, predict the most likely disease or condition. 
    Also provide home remedies and precautions. Format your response as JSON with the following structure:
    {{
        "disease": "name of the disease",
        "confidence": confidence percentage between 0 and 100,
        "remedies": ["remedy1", "remedy2", ...],
        "precautions": ["precaution1", "precaution2", ...]
    }}
    
    Symptoms: {symptoms}
    
    Provide only the JSON response without any additional text."""
def parse_gemini_response(response: str) -> Dict:
    """Parse the Gemini response and ensure it matches our expected format"""
    try:
        data = json.loads(response)
        required_keys = ["disease", "confidence", "remedies", "precautions"]
        if not all(key in data for key in required_keys):
            raise ValueError("Missing required keys in response")
        data["confidence"] = float(data["confidence"])
        if not 0 <= data["confidence"] <= 100:
            data["confidence"] = max(0, min(100, data["confidence"]))
            
        return data
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response from model")
@app.post("/predict", response_class=PlainTextResponse)  # Change response class to PlainTextResponse
async def predict_disease(input_data: SymptomsInput):
    try:
        prompt = generate_prompt(input_data.symptoms)
        response = model.generate_content(prompt)
        return response.text  # Return the raw text directly
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def root():
    return {"message": "Disease Prediction API is running. Send POST request to /predict endpoint with symptoms."}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
