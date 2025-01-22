# Disease Prediction API

This API uses Google's Gemini model to predict diseases based on symptoms. It provides a structured response with the predicted disease, confidence level, remedies, and precautions.

## Features
- Predicts the most likely disease or condition based on input symptoms.
- Provides home remedies and precautions.
- Returns a structured JSON response.

## Prerequisites
- Python 3.8 or higher
- `fastapi`, `uvicorn`, `pydantic`, and `google-generativeai` libraries installed
- A valid Google Generative AI API key

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/disease-prediction-api.git
   cd disease-prediction-api
   ```

2. Install the required Python libraries:
   ```bash
   pip install fastapi uvicorn pydantic google-generativeai
   ```

3. Set your Google Generative AI API key in the code or as an environment variable:
   ```python
   genai.configure(api_key='YOUR_GOOGLE_API_KEY')
   ```

## Usage
### Running the API
1. Start the FastAPI server:
   ```bash
   python main.py
   ```

2. The API will be available at `http://0.0.0.0:8000`.

### Endpoints
#### Root Endpoint
- **URL:** `GET /`
- **Description:** Provides a message confirming the API is running.
- **Response:**
  ```json
  {
    "message": "Disease Prediction API is running. Send POST request to /predict endpoint with symptoms."
  }
  ```

#### Predict Disease Endpoint
- **URL:** `POST /predict`
- **Description:** Predicts the disease based on symptoms.
- **Request Body:**
  ```json
  {
    "symptoms": "list your symptoms here"
  }
  ```
- **Response:**
  ```json
  {
    "disease": "Predicted disease name",
    "confidence": 85.5,
    "remedies": ["remedy1", "remedy2"],
    "precautions": ["precaution1", "precaution2"]
  }
  ```

### Example Usage
You can test the API using `curl` or tools like Postman.

#### Using `curl`
```bash
curl -X POST "http://0.0.0.0:8000/predict" \
-H "Content-Type: application/json" \
-d '{"symptoms": "fever, sore throat, fatigue"}'
```

#### Example Response
```json
{
  "disease": "Influenza",
  "confidence": 90.0,
  "remedies": ["Drink plenty of fluids", "Rest"],
  "precautions": ["Wash hands regularly", "Avoid close contact with infected individuals"]
}
```


