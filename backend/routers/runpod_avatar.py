import os
import json
import base64
import time
import uuid
import shutil
import asyncio
import requests
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from mutagen import File as MutagenFile
from dotenv import load_dotenv

from database import get_db
from auth_utils import get_current_user_id
from routers.billing import calculate_generation_cost, get_or_create_user

load_dotenv()

RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

RUNPOD_LIPSYNC_ENDPOINT_ID = os.getenv("RUNPOD_LIPSYNC_ENDPOINT_ID")
RUNPOD_LIPSYNC_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_LIPSYNC_ENDPOINT_ID}"

UNLIMITED_USER_IDS = {u.strip() for u in os.getenv("UNLIMITED_USER_IDS", "").split(",") if u.strip()}

TEMP_DIR = "data/runpod_tmp"
OUTPUT_DIR = "data/outputs"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

router = APIRouter(prefix="/api/generate", tags=["avatar-generation"])

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RUNPOD_API_KEY}"
}

LIPSYNC_PREFIX = "lipsync:"

# Кейс 3 с текстом (TTS -> липсинк): раньше TTS-шаг выполнялся
# синхронно внутри HTTP-запроса и иногда упирался в клиентский/прокси
# таймаут на холодном RunPod-воркере (см. историю чата — коды 499
# ровно на длительности клиентского таймаута). Теперь TTS + отправка
# в MuseTalk выполняются в фоне; TEXT_LIPSYNC_JOBS отслеживает статус
# такой фоновой задачи, пока она не превратится в обычный RunPod job_id
# с префиксом lipsync: (см. _run_lipsync_from_text_job и get_job_status).
TEXT_LIPSYNC_PREFIX = "textlipsync:"
TEXT_LIPSYNC_JOBS: dict[str, dict] = {}

JOB_OWNERS: dict[str, str] = {}


def get_media_duration_seconds(path: str) -> float:
    media = MutagenFile(path)
    if media is None or media.info is None or not hasattr(media.info, "length"):
        raise HTTPException(status_code=422, detail="Не удалось определить длительность файла")
    return float(media.info.length)


def charge_user(db: Session, user_id: str, cost: float):
    if user_id in UNLIMITED_USER_IDS:
        return
    user = get_or_create_user(db, user_id)
    if user.balance_usd < cost:
        raise HTTPException(
            status_code=402,
            detail=f"Недостаточно средств: нужно ${cost:.4f}, на балансе ${user.balance_usd:.4f}"
        )
    user.balance_usd -= cost
    db.commit()


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

    try:
        resp = requests.post(f"{RUNPOD_BASE_URL}/run", headers=HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Не удалось связаться с RunPod (SadTalker): {e}")

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

    try:
        resp = requests.post(f"{RUNPOD_BASE_URL}/run", headers=HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Не удалось связаться с RunPod (SadTalker/XTTS): {e}")

    job = resp.json()
    job_id = job.get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"RunPod не вернул id задачи: {job}")

    return job_id


def submit_lipsync_job(video_path: str, audio_path: str) -> str:
    if not RUNPOD_LIPSYNC_ENDPOINT_ID:
        raise HTTPException(
            status_code=500,
            detail="RUNPOD_LIPSYNC_ENDPOINT_ID не задан на сервере — проверь переменные окружения Railway"
        )

    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "video_base64": video_b64,
            "audio_base64": audio_b64,
        }
    }

    try:
        resp = requests.post(f"{RUNPOD_LIPSYNC_BASE_URL}/run", headers=HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Не удалось связаться с RunPod (lipsync): {e}")

    job = resp.json()
    raw_job_id = job.get("id")
    if not raw_job_id:
        raise HTTPException(status_code=502, detail=f"RunPod (lipsync) не вернул id задачи: {job}")

    return f"{LIPSYNC_PREFIX}{raw_job_id}"


def poll_runpod_job_sync(base_url: str, job_id: str, timeout: float = 120, interval: float = 3) -> dict:
    """Синхронно ждёт завершения RunPod-задачи (TTS) — вызывается из
    _run_lipsync_from_text_job через asyncio.to_thread, т.е. уже вне
    event loop, так что этот таймаут больше не рискует упереться ни в
    клиентский, ни в Railway-прокси таймаут — сам HTTP-запрос давно
    закончился к этому моменту."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{base_url}/status/{job_id}", headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "COMPLETED":
            return data
        if status == "FAILED":
            raise HTTPException(status_code=502, detail=f"Озвучка текста упала: {data}")
        time.sleep(interval)
    raise HTTPException(status_code=504, detail="Озвучка текста не успела завершиться вовремя")


def synthesize_tts_via_runpod(voice_sample_path: str, text: str, language: str, work_dir: str) -> tuple[str, float]:
    with open(voice_sample_path, "rb") as f:
        voice_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "tts_only": True,
            "text": text,
            "voice_sample_base64": voice_b64,
            "language": language,
        }
    }
    try:
        resp = requests.post(f"{RUNPOD_BASE_URL}/run", headers=HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Не удалось запустить озвучку текста: {e}")

    job = resp.json()
    job_id = job.get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"RunPod не вернул id задачи озвучки: {job}")

    result = poll_runpod_job_sync(RUNPOD_BASE_URL, job_id)
    audio_b64 = result.get("output", {}).get("audio_base64")
    if not audio_b64:
        raise HTTPException(status_code=502, detail=f"Озвучка не вернула audio_base64: {result}")

    audio_path = os.path.join(work_dir, "tts_output.wav")
    with open(audio_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    duration = get_media_duration_seconds(audio_path)
    return audio_path, duration


async def _run_lipsync_from_text_job(internal_id: str, video_path: str, voice_path: str,
                                      work_dir: str, text: str, language: str, user_id: str):
    """Фоновая обработка Кейса 3 с текстом — TTS и отправка в MuseTalk,
    полностью вне HTTP-запроса. Регистрирует результат в JOB_OWNERS
    (тот же механизм отложенного списания, что уже используется в
    Кейсе 1) — списание произойдёт при получении готового видео в
    get_job_status, как обычно."""
    try:
        audio_path, duration = await asyncio.to_thread(
            synthesize_tts_via_runpod, voice_path, text, language, work_dir
        )
        runpod_job_id = await asyncio.to_thread(submit_lipsync_job, video_path, audio_path)
        JOB_OWNERS[runpod_job_id] = user_id
        TEXT_LIPSYNC_JOBS[internal_id] = {"status": "SUBMITTED", "runpod_job_id": runpod_job_id, "error": None}
    except HTTPException as e:
        TEXT_LIPSYNC_JOBS[internal_id] = {"status": "FAILED", "runpod_job_id": None, "error": str(e.detail)}
    except Exception as e:
        TEXT_LIPSYNC_JOBS[internal_id] = {"status": "FAILED", "runpod_job_id": None, "error": str(e)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@router.post("/photo-emotion")
async def generate_photo_emotion(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    expression_scale: float = Form(0.7),
    pose_style: int = Form(0),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    request_id = uuid.uuid4().hex
    image_path = os.path.join(TEMP_DIR, f"{request_id}_{image.filename}")
    audio_path = os.path.join(TEMP_DIR, f"{request_id}_{audio.filename}")

    with open(image_path, "wb") as f:
        f.write(await image.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    try:
        duration = get_media_duration_seconds(audio_path)
        cost = calculate_generation_cost(duration)
        charge_user(db, user_id, cost)

        job_id = await asyncio.to_thread(
            submit_sadtalker_job,
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
    language: str = Form("ru"),
    user_id: str = Depends(get_current_user_id),
):
    request_id = uuid.uuid4().hex
    image_path = os.path.join(TEMP_DIR, f"{request_id}_{image.filename}")
    voice_path = os.path.join(TEMP_DIR, f"{request_id}_{voice_sample.filename}")

    with open(image_path, "wb") as f:
        f.write(await image.read())
    with open(voice_path, "wb") as f:
        f.write(await voice_sample.read())

    try:
        job_id = await asyncio.to_thread(
            submit_photo_text_emotion_job,
            image_path, voice_path, text, emotion, language
        )
    finally:
        for p in (image_path, voice_path):
            if os.path.exists(p):
                os.remove(p)

    JOB_OWNERS[job_id] = user_id
    return {"job_id": job_id, "status_url": f"/api/generate/status/{job_id}"}


@router.post("/lipsync")
async def generate_lipsync(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    request_id = uuid.uuid4().hex
    video_path = os.path.join(TEMP_DIR, f"{request_id}_{video.filename}")
    audio_path = os.path.join(TEMP_DIR, f"{request_id}_{audio.filename}")

    with open(video_path, "wb") as f:
        f.write(await video.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    try:
        duration = get_media_duration_seconds(audio_path)
        cost = calculate_generation_cost(duration)
        charge_user(db, user_id, cost)

        job_id = await asyncio.to_thread(submit_lipsync_job, video_path, audio_path)
    finally:
        for p in (video_path, audio_path):
            if os.path.exists(p):
                os.remove(p)

    return {"job_id": job_id, "status_url": f"/api/generate/status/{job_id}"}


@router.post("/lipsync-from-text")
async def generate_lipsync_from_text(
    video: UploadFile = File(...),
    voice_sample: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form("ru"),
    user_id: str = Depends(get_current_user_id),
):
    """Кейс 3 с текстом вместо готового аудио: озвучивает текст через
    SadTalker/XTTS-воркер (tts_only), затем отправляет липсинк-задачу
    в MuseTalk.

    ВАЖНО: переписано под тот же submit→poll паттерн, что и все
    остальные эндпоинты — раньше TTS-шаг выполнялся синхронно внутри
    этого HTTP-запроса, из-за чего на холодном/перегруженном
    SadTalker/XTTS-воркере суммарное время иногда превышало и наш
    собственный клиентский таймаут, и рисковало упереться в таймаут
    прокси Railway — оба раза с одной и той же картиной: запрос
    обрывается уже после того, как всё фактически успешно случилось
    на сервере (см. историю чата: коды 499 "client closed request"
    ровно на длительности клиентского таймаута — это наш же frontend,
    а не сетевой сбой).

    Теперь TTS и отправка задачи выполняются в фоне — сам HTTP-запрос
    возвращается за доли секунды с временным job_id (префикс
    textlipsync:). Списание баланса теперь ОТЛОЖЕНО до готового видео
    (тот же механизм JOB_OWNERS, что уже в Кейсе 1) — раньше
    списывалось сразу после TTS-шага, ДО завершения HTTP-запроса."""
    request_id = uuid.uuid4().hex
    work_dir = os.path.join(TEMP_DIR, request_id)
    os.makedirs(work_dir, exist_ok=True)
    video_path = os.path.join(work_dir, f"video_{video.filename}")
    voice_path = os.path.join(work_dir, f"voice_{voice_sample.filename}")

    with open(video_path, "wb") as f:
        f.write(await video.read())
    with open(voice_path, "wb") as f:
        f.write(await voice_sample.read())

    internal_id = uuid.uuid4().hex
    TEXT_LIPSYNC_JOBS[internal_id] = {"status": "PENDING", "runpod_job_id": None, "error": None}

    asyncio.create_task(
        _run_lipsync_from_text_job(internal_id, video_path, voice_path, work_dir, text, language, user_id)
    )

    job_id = f"{TEXT_LIPSYNC_PREFIX}{internal_id}"
    return {"job_id": job_id, "status_url": f"/api/generate/status/{job_id}"}


@router.get("/status/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Опрашивается клиентом до тех пор, пока генерация не завершится.

    job_id с префиксом "textlipsync:" — фоновая задача Кейса 3 с
    текстом, ещё не превратившаяся в реальный RunPod job_id (см.
    _run_lipsync_from_text_job); "lipsync:" — реальный эндпоинт Кейса 3
    (MuseTalk); всё остальное — основной эндпоинт (SadTalker/XTTS,
    Кейсы 1 и 2)."""
    if job_id.startswith(TEXT_LIPSYNC_PREFIX):
        internal_id = job_id[len(TEXT_LIPSYNC_PREFIX):]
        entry = TEXT_LIPSYNC_JOBS.get(internal_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Задача не найдена (возможно, сервер перезапускался)")
        if entry["status"] == "PENDING":
            return {"job_id": job_id, "status": "PENDING"}
        if entry["status"] == "FAILED":
            raise HTTPException(status_code=502, detail=f"Генерация упала: {entry['error']}")
        # SUBMITTED — дальше работаем с реальным RunPod job_id как обычно.
        # Запись НЕ удаляем из TEXT_LIPSYNC_JOBS: клиент продолжает
        # опрашивать этот же исходный job_id на каждом последующем
        # запросе, так что резолвить префикс нужно уметь многократно.
        job_id = entry["runpod_job_id"]

    if job_id.startswith(LIPSYNC_PREFIX):
        raw_job_id = job_id[len(LIPSYNC_PREFIX):]
        base_url = RUNPOD_LIPSYNC_BASE_URL
    else:
        raw_job_id = job_id
        base_url = RUNPOD_BASE_URL

    status_url = f"{base_url}/status/{raw_job_id}"
    status_resp = await asyncio.to_thread(requests.get, status_url, headers=HEADERS, timeout=30)
    data = status_resp.json()
    status = data.get("status")

    if status == "COMPLETED":
        video_b64 = data["output"]["video_base64"]
        out_path = os.path.join(OUTPUT_DIR, f"{raw_job_id}.mp4")
        first_time = not os.path.exists(out_path)

        if first_time:
            def _write_video():
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(video_b64))

            await asyncio.to_thread(_write_video)

            owner_user_id = JOB_OWNERS.pop(job_id, None)
            if owner_user_id:
                try:
                    actual_duration = get_media_duration_seconds(out_path)
                    cost = calculate_generation_cost(actual_duration)
                    charge_user(db, owner_user_id, cost)
                except HTTPException:
                    raise
                except Exception as e:
                    print(f"Не удалось списать средства за задачу {job_id}: {e}")

        return FileResponse(out_path, media_type="video/mp4", filename="avatar_result.mp4")

    if status == "FAILED":
        raise HTTPException(status_code=502, detail=f"Генерация упала: {data}")

    return {"job_id": job_id, "status": status}
