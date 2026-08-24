import os
import json
import base64
import time
import uuid
import requests
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()

RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

TEMP_DIR = "data/runpod_tmp"
OUTPUT_DIR = "data/outputs"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

router = APIRouter(prefix="/api/generate", tags=["avatar-generation"])

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RUNPOD_API_KEY}"
}


# ---------------------------------------------------------------------------
# Важно: эндпоинты FastAPI ниже НЕ ждут завершения генерации внутри одного
# HTTP-запроса. Railway (edge-прокси перед приложением) обрывает долгие
# запросы своим собственным таймаутом независимо от кода приложения —
# генерация видео (SadTalker + XTTS-v2) может занимать 5+ минут, что
# превышает этот лимит. Поэтому используется схема submit → poll:
#   1) POST /photo-text-emotion (или /photo-emotion) сразу отправляет
#      задачу в RunPod и возвращает job_id — сам HTTP-запрос занимает
#      секунды.
#   2) Клиент (или тестировщик через Swagger/curl) опрашивает
#      GET /status/{job_id} с любым интервалом, пока не получит готовое
#      видео — каждый такой запрос тоже быстрый, так что Railway его не
#      обрывает.
# ---------------------------------------------------------------------------


def submit_sadtalker_job(image_path: str, audio_path: str, expression_scale: float,
                          pose_style: int, size: int = 512, still: bool = True,
                          enhancer: str = "gfpgan") -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "image_base64": img_b64,
            "audio_base64": audio_b64,
            "expression_scale": expression_scale,
            "pose_style": pose_style,
            "size": size,
            "still": still,
            "enhancer": enhancer
        }
    }

    resp = requests.post(f"{RUNPOD_BASE_URL}/run", headers=HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    job = resp.json()
    job_id = job.get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"RunPod не вернул id задачи: {job}")

    return job_id


def submit_photo_text_emotion_job(image_path: str, voice_sample_path: str, text: str,
                                   emotion: str, language: str = "ru") -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(voice_sample_path, "rb") as f:
        voice_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "image_base64": img_b64,
            "voice_sample_base64": voice_b64,
            "text": text,
            "emotion": emotion,
            "language": language,
        }
    }

    resp = requests.post(f"{RUNPOD_BASE_URL}/run", headers=HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    job = resp.json()
    job_id = job.get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"RunPod не вернул id задачи: {job}")

    return job_id


@router.post("/photo-emotion")
async def generate_photo_emotion(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    expression_scale: float = Form(0.7),
    pose_style: int = Form(0)
):
    """Кейс 2: фото + аудио, модель сама подстраивает эмоции под голос.
    Сразу возвращает job_id — результат забирается через GET /status/{job_id}."""
    request_id = uuid.uuid4().hex
    image_path = os.path.join(TEMP_DIR, f"{request_id}_{image.filename}")
    audio_path = os.path.join(TEMP_DIR, f"{request_id}_{audio.filename}")

    with open(image_path, "wb") as f:
        f.write(await image.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    try:
        job_id = submit_sadtalker_job(
            image_path, audio_path,
            expression_scale=expression_scale,
            pose_style=pose_style
        )
    finally:
        for p in (image_path, audio_path):
            if os.path.exists(p):
                os.remove(p)

    return {"job_id": job_id, "status_url": f"/api/generate/status/{job_id}"}


@router.post("/photo-text-emotion")
async def generate_photo_text_emotion(
    image: UploadFile = File(...),
    voice_sample: UploadFile = File(...),
    text: str = Form(...),
    emotion: str = Form("neutral"),
    language: str = Form("ru")
):
    """Кейс 1: фото + образец голоса + текст + выбор эмоции.
    Сразу возвращает job_id — результат забирается через GET /status/{job_id}."""
    request_id = uuid.uuid4().hex
    image_path = os.path.join(TEMP_DIR, f"{request_id}_{image.filename}")
    voice_path = os.path.join(TEMP_DIR, f"{request_id}_{voice_sample.filename}")

    with open(image_path, "wb") as f:
        f.write(await image.read())
    with open(voice_path, "wb") as f:
        f.write(await voice_sample.read())

    try:
        job_id = submit_photo_text_emotion_job(
            image_path, voice_path, text, emotion, language
        )
    finally:
        for p in (image_path, voice_path):
            if os.path.exists(p):
                os.remove(p)

    return {"job_id": job_id, "status_url": f"/api/generate/status/{job_id}"}


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Опрашивается клиентом (или вручную через Swagger/curl) до тех пор,
    пока генерация не завершится. Быстрый запрос — не подвержен таймауту
    Railway edge-прокси, в отличие от прямого ожидания результата."""
    status_url = f"{RUNPOD_BASE_URL}/status/{job_id}"
    status_resp = requests.get(status_url, headers=HEADERS, timeout=30)
    data = status_resp.json()
    status = data.get("status")

    if status == "COMPLETED":
        video_b64 = data["output"]["video_base64"]
        out_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(video_b64))
        return FileResponse(out_path, media_type="video/mp4", filename="avatar_result.mp4")

    if status == "FAILED":
        raise HTTPException(status_code=502, detail=f"Генерация упала: {data}")

    # IN_QUEUE / IN_PROGRESS и т.п. — сообщаем клиенту, что нужно спросить позже
    return {"job_id": job_id, "status": status}
