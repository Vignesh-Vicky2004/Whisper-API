import whisper
import tempfile
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Whisper API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once globally
model = None

@app.on_event("startup")
async def startup_event():
    global model
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # Start with base model for faster startup
    print("Model loaded successfully!")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        if model is None:
            return JSONResponse(status_code=503, content={"error": "Model not loaded yet"})
        
        # Save the uploaded audio to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name
        
        # Transcribe
        result = model.transcribe(temp_audio_path)
        
        # Cleanup
        os.remove(temp_audio_path)
        
        return {"text": result["text"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "Whisper API is running", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/ready")
def ready():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "not ready", "model_loaded": False})
    return {"status": "ready", "model_loaded": True}
