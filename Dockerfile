FROM python:3.10

# Enable ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

# Install CUDA 11.8 compatible torch
RUN pip install --upgrade pip \
 && pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu118

# Install remaining deps
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "whisper_api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120", "--lifespan", "on"]
