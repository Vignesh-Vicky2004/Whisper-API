FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y ffmpeg git curl python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install openai-whisper fastapi uvicorn

WORKDIR /app
COPY whisper_api.py .

CMD ["uvicorn", "whisper_api:app", "--host", "0.0.0.0", "--port", "8000"]
