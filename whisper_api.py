import whisper
import tempfile
import os
import torch
import subprocess
import ffmpeg
from fastapi import FastAPI, UploadFile, File, HTTPException
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

# Global model variable
model = None
device = None

# Supported audio formats
SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.opus', '.webm', '.mp4', '.mov', '.avi'}

@app.on_event("startup")
async def startup_event():
    global model, device
    print("Checking CUDA availability...")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    print(f"Loading Whisper model on {device}...")
    try:
        # Use base model for faster startup, can be changed to medium/large for better accuracy
        model = whisper.load_model("base", device=device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def validate_audio_file(file_path: str) -> bool:
    """Validate if the file is a proper audio file using ffprobe"""
    try:
        probe = ffmpeg.probe(file_path)
        # Check if there's at least one audio stream
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        return len(audio_streams) > 0
    except Exception as e:
        print(f"Audio validation failed: {e}")
        return False

def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert audio file to WAV format using ffmpeg"""
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        print(f"FFmpeg conversion error: {e.stderr.decode()}")
        return False

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_files = []  # Keep track of temp files for cleanup
    
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        
        # Check file size (limit to 100MB)
        content = await file.read()
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Max size: 100MB")
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Get file extension
        file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        if not file_extension:
            file_extension = ".wav"  # Default fallback
        
        # Check if format is supported
        if file_extension not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format: {file_extension}")
        
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_input:
            temp_input.write(content)
            temp_input_path = temp_input.name
            temp_files.append(temp_input_path)
        
        print(f"Processing audio file: {file.filename} ({len(content)} bytes) with extension: {file_extension}")
        
        # Validate the audio file
        if not validate_audio_file(temp_input_path):
            raise HTTPException(status_code=400, detail="Invalid audio file or corrupted data")
        
        # Convert to WAV if needed
        if file_extension != '.wav':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                temp_wav_path = temp_wav.name
                temp_files.append(temp_wav_path)
            
            print(f"Converting {file_extension} to WAV...")
            if not convert_to_wav(temp_input_path, temp_wav_path):
                raise HTTPException(status_code=500, detail="Failed to convert audio file")
            
            audio_path = temp_wav_path
        else:
            audio_path = temp_input_path
        
        # Transcribe with GPU acceleration
        print("Starting transcription...")
        result = model.transcribe(
            audio_path, 
            fp16=device=="cuda",
            verbose=False,
            word_timestamps=False
        )
        
        print("Transcription completed successfully")
        
        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "device_used": device,
            "file_info": {
                "filename": file.filename,
                "size_bytes": len(content),
                "format": file_extension,
                "converted": file_extension != '.wav'
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")

@app.get("/")
def root():
    return {
        "message": "Whisper API is running", 
        "status": "healthy",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "supported_formats": list(SUPPORTED_FORMATS)
    }

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/ready")
def ready():
    if model is None:
        return JSONResponse(status_code=503, content={
            "status": "not ready", 
            "model_loaded": False,
            "device": device
        })
    return {
        "status": "ready", 
        "model_loaded": True,
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/info")
def info():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f} GB",
            "gpu_memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**3:.1f} GB"
        }
    
    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "supported_formats": list(SUPPORTED_FORMATS),
        **gpu_info
    }

@app.post("/transcribe-with-options")
async def transcribe_audio_with_options(
    file: UploadFile = File(...),
    model_size: str = "base",
    language: str = None,
    word_timestamps: bool = False
):
    """Advanced transcription with more options"""
    # This would require loading different model sizes
    # For now, just use the default model with additional options
    temp_files = []
    
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        
        # Similar file processing as above...
        content = await file.read()
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Max size: 100MB")
        
        file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ".wav"
        if file_extension not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format: {file_extension}")
        
        # Save and process file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_input:
            temp_input.write(content)
            temp_input_path = temp_input.name
            temp_files.append(temp_input_path)
        
        if not validate_audio_file(temp_input_path):
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        # Convert if needed
        if file_extension != '.wav':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                temp_wav_path = temp_wav.name
                temp_files.append(temp_wav_path)
            
            if not convert_to_wav(temp_input_path, temp_wav_path):
                raise HTTPException(status_code=500, detail="Failed to convert audio file")
            audio_path = temp_wav_path
        else:
            audio_path = temp_input_path
        
        # Transcribe with options
        transcribe_options = {
            "fp16": device == "cuda",
            "verbose": False,
            "word_timestamps": word_timestamps
        }
        
        if language:
            transcribe_options["language"] = language
        
        result = model.transcribe(audio_path, **transcribe_options)
        
        response = {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "device_used": device,
            "options_used": transcribe_options
        }
        
        if word_timestamps and "segments" in result:
            response["segments"] = result["segments"]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Advanced transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")
