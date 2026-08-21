"""
Avatar Studio — локальный бэкенд-оркестратор.

Запускается на машине пользователя (localhost), обслуживает Electron-UI.
Ничего не уходит в облако — вся генерация локальная, поэтому она бесплатна
для всех: нет ни серверных GPU-расходов на вашей стороне, ни платы за
API-запросы на стороне пользователя.

Запуск: uvicorn main:app --port 8420
"""
from __future__ import annotations

import asyncio
import uuid
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import avatar, voice_clone, lipsync, stitch

app = FastAPI(title="Avatar Studio Backend")

# Electron-фронтенд обращается с localhost на другом порту — разрешаем.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "app://."],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path("data/uploads")
PROJECTS_DIR = Path("data/projects")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job tracking для MVP. Для продакшна — заменить на SQLite
# (sqlalchemy уже в requirements.txt) с сохранением между перезапусками.
JOBS: dict[str, lipsync.LipsyncJob] = {}


@app.post("/avatar/create")
async def create_avatar(
    name: str = Form(...),
    consent_confirmed: bool = Form(...),
    file: UploadFile = File(...),
):
    if not consent_confirmed:
        raise HTTPException(
            400,
            "Нужно подтвердить, что это ваше лицо или у вас есть право "
            "его использовать (consent_confirmed).",
        )
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    dest.write_bytes(await file.read())

    try:
        profile = avatar.create_avatar_profile(
            source_media_path=str(dest),
            name=name,
            consent_confirmed=consent_confirmed,
        )
    except avatar.NoFaceDetectedError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ошибка создания аватара: {e}")
    return profile.to_json()


@app.post("/voice/clone")
async def create_voice(
    name: str = Form(...),
    consent_confirmed: bool = Form(...),
    file: UploadFile = File(...),
):
    if not consent_confirmed:
        raise HTTPException(
            400,
            "Нужно подтвердить согласие на клонирование голоса "
            "(consent_confirmed).",
        )
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    dest.write_bytes(await file.read())

    profile = voice_clone.clone_voice(
        sample_audio_path=str(dest),
        name=name,
        consent_confirmed=consent_confirmed,
    )
    return profile.to_json()


class EmotionPreset(str, Enum):
    """
    Пресеты эмоций для сегмента. Каждому (кроме fallback-сценария)
    соответствует готовый видео-драйвер в data/emotion_drivers/*.mp4,
    который используется как video_path для MuseTalk вместо статичного
    опорного кадра аватара — см. pipeline/lipsync.resolve_video_source().
    """
    neutral = "neutral"
    happy = "happy"
    sadness = "sadness"
    anger = "anger"
    love = "love"  # видео-драйвер пока не записан — см. GET /emotions


class GenerateSegmentRequest(BaseModel):
    avatar_id: str
    voice_id: str
    text: str
    language: str = "ru"
    emotion: EmotionPreset = EmotionPreset.neutral


@app.post("/segment/generate")
async def generate_segment(req: GenerateSegmentRequest):
    """
    Полный цикл одного сегмента: текст -> клонированная речь -> липсинк.
    До 5 минут на сегмент (лимит зашит в pipeline.lipsync.MAX_SEGMENT_SECONDS).

    Синтез речи выполняется здесь же (обычно быстрый), а сам липсинк-
    инференс — тяжёлая и долгая операция (минуты) — запускается в
    фоновом потоке через asyncio.to_thread, чтобы этот HTTP-запрос
    отвечал почти мгновенно с job_id. Фронтенд узнаёт результат через
    короткие опросы GET /jobs/{id} (см. waitForJob в src/app.js).
    """
    avatar.load_avatar_profile(req.avatar_id)  # бросит понятную ошибку, если нет

    try:
        audio_path = voice_clone.synthesize(req.text, req.voice_id, req.language)
    except NotImplementedError as e:
        raise HTTPException(501, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        # Печатаем полный трейсбек в консоль сервера — короткое сообщение
        # в HTTP-ответе часто теряет важные детали (файл, строку, точную
        # причину), а именно они нужны при отладке на новой машине.
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Ошибка синтеза речи: {e}")

    job = lipsync.create_job(req.avatar_id, str(audio_path), emotion=req.emotion.value)
    JOBS[job.job_id] = job

    # Генерация может занимать несколько минут — запускаем её в фоновом
    # потоке и сразу отвечаем клиенту job_id, чтобы не держать HTTP-запрос
    # открытым на всё время инференса.
    asyncio.create_task(asyncio.to_thread(lipsync.run_job, job))

    return {
        "job_id": job.job_id,
        "status": job.status,
        "chunk_count": job.chunk_count,
        "total_duration_sec": job.total_duration_sec,
        "emotion": job.emotion,
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    return job


@app.get("/emotions")
async def list_emotions():
    """
    Список пресетов эмоций с флагом реальной доступности (существует ли
    файл видео-драйвера на диске). Фронтенд использует это, чтобы не
    предлагать пользователю эмоции, для которых ещё нет записанного видео
    (например, love.mp4), не дожидаясь ошибки на этапе генерации.
    """
    return {"emotions": lipsync.list_available_emotions()}


class StitchRequest(BaseModel):
    project_id: str
    segment_paths: list[str]
    transition_sec: float = 0.6


@app.post("/project/stitch")
async def stitch_project(req: StitchRequest):
    """Собирает несколько готовых сегментов в один фильм с кроссфейдом."""
    out_dir = PROJECTS_DIR / req.project_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / "final_film.mp4")

    stitch.stitch_segments(req.segment_paths, output_path, req.transition_sec)
    manifest = stitch.build_project_manifest(
        req.project_id, req.segment_paths, str(out_dir)
    )
    return {"output_path": output_path, "manifest": manifest}


@app.get("/health")
async def health():
    return {"status": "ok"}