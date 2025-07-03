import whisper
import tempfile
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

# Load the model once globally
model = whisper.load_model("large")  # You can use "base", "medium", etc., if GPU/space is limited

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
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
