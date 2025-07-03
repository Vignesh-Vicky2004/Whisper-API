import whisper
from fastapi import FastAPI, UploadFile, File

app = FastAPI()
model = whisper.load_model("large")  # Choose "large" for best accuracy

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    contents = await file.read()
    with open("audio.wav", "wb") as f:
        f.write(contents)
    result = model.transcribe("audio.wav")
    return {"text": result["text"]}
