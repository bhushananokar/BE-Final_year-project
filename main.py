import os
import base64
import uvicorn
import re
import tempfile
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional

# Configure Gemini
genai.configure(api_key='AIzaSyAuU21y64bm80r-5mxq2IUbBH1VKd3sZ28')
text_model = genai.GenerativeModel('gemini-pro')

# Generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

app = FastAPI(
    title="Health API",
    description="API for disease prediction, health recommendations, blood report analysis, and disease image analysis"
)

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

class SymptomsInput(BaseModel):
    symptoms: str

class UserProfile(BaseModel):
    age: int = Field(..., gt=0, lt=150)
    height: float = Field(..., description="Height in centimeters")
    weight: float = Field(..., description="Weight in kilograms")
    blood_group: str
    gender: str
    activity_level: str
    existing_conditions: List[str] = Field(default=[])
    medications: List[str] = Field(default=[])

    @field_validator('blood_group')
    def validate_blood_group(cls, v):
        if not re.match(r'^(A|B|AB|O)[+-]$', v):
            raise ValueError('Invalid blood group. Must be A+/-, B+/-, AB+/-, or O+/-')
        return v

    @field_validator('gender')
    def validate_gender(cls, v):
        if v.lower() not in ['male', 'female', 'other']:
            raise ValueError('Gender must be male, female, or other')
        return v.lower()

    @field_validator('activity_level')
    def validate_activity_level(cls, v):
        valid_levels = ['sedentary', 'light', 'moderate', 'active', 'very_active']
        if v.lower() not in valid_levels:
            raise ValueError(f'Activity level must be one of: {", ".join(valid_levels)}')
        return v.lower()

def calculate_bmi(weight: float, height: float) -> float:
    height_in_meters = height / 100
    return round(weight / (height_in_meters ** 2), 2)

def generate_disease_prompt(symptoms: str) -> str:
    return f"""Based on the following symptoms, predict the most likely disease or condition. 
    Provide information in this structure:
    1. Most likely disease/condition
    2. Confidence level
    3. Home remedies (list at least 3)
    4. Precautions to take (list at least 3)
    
    Symptoms: {symptoms}"""

def generate_health_prompt(user_data: UserProfile) -> str:
    bmi = calculate_bmi(user_data.weight, user_data.height)
    
    return f"""Based on the following user profile, provide detailed personalized health recommendations.
    
    User Profile:
    - Age: {user_data.age}
    - Gender: {user_data.gender}
    - BMI: {bmi}
    - Blood Group: {user_data.blood_group}
    - Activity Level: {user_data.activity_level}
    - Existing Conditions: {', '.join(user_data.existing_conditions) if user_data.existing_conditions else 'None'}
    - Current Medications: {', '.join(user_data.medications) if user_data.medications else 'None'}

    Please provide:
    1. Fitness Recommendations (include specific exercises with duration and intensity)
    2. Dietary Recommendations (include specific foods and meal timing)
    3. Lifestyle Recommendations
    4. Important Health Alerts or Precautions"""

def generate_blood_report_prompt() -> str:
    return """Analyze this blood test report image and provide:

    1. List each parameter with its:
       - Measured value
       - Normal range
       - Whether it's normal, high, or low

    2. Key findings from the report

    3. Recommendations based on the results

    Please provide the analysis in a clear, readable format."""

def generate_disease_image_prompt() -> str:
    return """Analyze this medical image (which may be an X-ray, CT scan, MRI, or other medical imaging) and provide:

    1. Predicted disease or condition
    2. Severity level (Mild, Moderate, Severe, or Critical)
    3. Confidence in the diagnosis (percentage)
    4. Key findings visible in the image
    5. Recommendations based on the findings

    Format your response as follows:
    DISEASE: [disease name]
    SEVERITY: [severity level]
    CONFIDENCE: [confidence percentage]
    FINDINGS: [list key observations from the image]
    RECOMMENDATIONS: [list recommendations]"""

@app.post("/predict", response_class=PlainTextResponse)
async def predict_disease(input_data: SymptomsInput):
    try:
        prompt = generate_disease_prompt(input_data.symptoms)
        response = text_model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/health-recommendations", response_class=PlainTextResponse)
async def get_health_recommendations(user_data: UserProfile):
    try:
        prompt = generate_health_prompt(user_data)
        response = text_model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report", response_class=PlainTextResponse)
async def analyze_blood_report(file: UploadFile = File(...)):
    try:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Upload the file to Gemini
        uploaded_file = upload_to_gemini(temp_file_path, mime_type=file.content_type)

        # Start a chat session with the uploaded file
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [uploaded_file],
                },
            ]
        )

        # Generate content analysis
        response = chat_session.send_message(generate_blood_report_prompt())

        # Clean up the temporary file
        os.unlink(temp_file_path)

        return response.text

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-disease-image", response_class=PlainTextResponse)
async def analyze_disease_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        valid_mime_types = ["image/jpeg", "image/png", "image/dicom", "image/tiff"]
        if file.content_type not in valid_mime_types and not file.content_type.startswith("application/dicom"):
            raise HTTPException(status_code=400, detail=f"Invalid file type. Supported types: {', '.join(valid_mime_types)}")

        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Upload the file to Gemini
        uploaded_file = upload_to_gemini(temp_file_path, mime_type=file.content_type)

        # Start a chat session with the uploaded file
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [uploaded_file],
                },
            ]
        )

        # Generate disease analysis
        response = chat_session.send_message(generate_disease_image_prompt())

        # Clean up the temporary file
        os.unlink(temp_file_path)

        return response.text

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Health API is running. Available endpoints:",
        "endpoints": {
            "/predict": "Disease prediction based on symptoms",
            "/health-recommendations": "Personalized health recommendations",
            "/report": "Blood report analysis from image",
            "/analyze-disease-image": "Disease analysis from medical images including X-rays"
        },
        "example_requests": {
            "predict": {
                "symptoms": "fever, headache, fatigue"
            },
            "health-recommendations": {
                "age": 30,
                "height": 170,
                "weight": 70,
                "blood_group": "O+",
                "gender": "female",
                "activity_level": "moderate",
                "existing_conditions": ["asthma"],
                "medications": ["albuterol"]
            },
            "report": "POST multipart/form-data with blood report image file",
            "analyze-disease-image": "POST multipart/form-data with medical image file (X-ray, CT scan, etc.)"
        }
    }

#if __name__ == "__main__":
#   uvicorn.run(app, host="0.0.0.0", port=8000)
