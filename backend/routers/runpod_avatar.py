import os
import json
import base64
import time
import uuid
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

# Кейс 3 (MuseTalk-липсинк) живёт на отдельном RunPod Serverless
# эндпоинте (Avatar-Studio-Lipsync), отдельном от SadTalker/XTTS
# (Avatar-Studio). API-ключ общий, endpoint ID — свой.
RUNPOD_LIPSYNC_ENDPOINT_ID = os.getenv("RUNPOD_LIPSYNC_ENDPOINT_ID")
RUNPOD_LIPSYNC_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_LIPSYNC_ENDPOINT_ID}"

# Список user_id (из Supabase, тот же формат, что отдаёт
# GET /api/billing/balance в поле "user_id"), для которых генерация
# НЕ списывает баланс — например, твой собственный аккаунт для
# тестирования. Задаётся через Railway Variables, через запятую:
# UNLIMITED_USER_IDS=abc-123-...,def-456-...
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

# Префикс, которым помечаются job_id, отправленные на lipsync-эндпоинт,
# чтобы GET /status/{job_id} знал, к какому воркеру стучаться.
LIPSYNC_PREFIX = "lipsync:"

# Кейс 1 (текст -> TTS -> видео): длительность озвучки известна только
# после реальной генерации, поэтому списание баланса откладывается до
# момента получения готового видео в /status/{job_id}. Здесь храним,
# какому пользователю принадлежит задача. ВАЖНО: это простое решение
# "в памяти процесса" — при рестарте/редеплое Railway между отправкой
# задачи и её завершением запись потеряется и списание для этой
# конкретной задачи не произойдёт (редкий, но возможный случай).
JOB_OWNERS: dict[str, str] = {}


def get_media_duration_seconds(path: str) -> float:
    """Читает реальную длительность аудио/видео файла на диске через
    mutagen — сервер не доверяет числам, которые прислал клиент."""
    media = MutagenFile(path)
    if media is None or media.info is None or not hasattr(media.info, "length"):
        raise HTTPException(status_code=422, detail="Не удалось определить длительность файла")
    return float(media.info.length)


def charge_user(db: Session, user_id: str, cost: float):
    """Списывает cost с баланса пользователя, если он не в списке
    освобождённых от оплаты. Бросает 402, если средств не хватает."""
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


# ---------------------------------------------------------------------------
# Важно: эндпоинты FastAPI ниже НЕ ждут завершения генерации внутри одного
# HTTP-запроса. Railway (edge-прокси перед приложением) обрывает долгие
# запросы своим собственным таймаутом независимо от кода приложения —
# генерация видео (SadTalker + XTTS-v2 / MuseTalk) может занимать 5+ минут,
# что превышает этот лимит. Поэтому используется схема submit → poll:
#   1) POST /photo-text-emotion (или /photo-emotion, /lipsync) сразу
#      отправляет задачу в RunPod и возвращает job_id — сам HTTP-запрос
#      занимает секунды.
#   2) Клиент опрашивает GET /status/{job_id} с любым интервалом, пока
#      не получит готовое видео — каждый такой запрос тоже быстрый, так
#      что Railway его не обрывает.
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
    """Кейс 3: отдельный RunPod-эндпоинт (MuseTalk)."""
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


@router.post("/photo-emotion")
async def generate_photo_emotion(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    expression_scale: float = Form(0.7),
    pose_style: int = Form(0),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Кейс 2: фото + аудио, модель сама подстраивает эмоции под голос.
    Списывает баланс сразу — длительность известна по загруженному
    аудио. Сразу возвращает job_id — результат забирается через
    GET /status/{job_id}."""
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
    """Кейс 1: фото + образец голоса + текст + выбор эмоции. Длительность
    озвучки известна только после реального TTS, поэтому списание
    баланса откладывается до момента, когда видео будет готово (см.
    /status/{job_id}). Сразу возвращает job_id."""
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
    """Кейс 3: видео с лицом + аудио-драйвер → липсинк через MuseTalk 1.5.
    Списывает баланс сразу — длительность известна по загруженному
    аудио-драйверу."""
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


@router.get("/status/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Опрашивается клиентом до тех пор, пока генерация не завершится.

    job_id с префиксом "lipsync:" направляется на RunPod-эндпоинт Кейса 3
    (MuseTalk), все остальные — на основной эндпоинт (SadTalker/XTTS,
    Кейсы 1 и 2)."""
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
        # Идемпотентность: если файл уже записан (повторный опрос статуса
        # после завершения), не перезаписываем и, что важнее, НЕ списываем
        # баланс повторно за ту же задачу.
        first_time = not os.path.exists(out_path)

        if first_time:
            def _write_video():
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(video_b64))

            await asyncio.to_thread(_write_video)

            # Отложенное списание для Кейса 1 (текст -> TTS) — длительность
            # известна только сейчас, по факту готового видео.
            owner_user_id = JOB_OWNERS.pop(job_id, None)
            if owner_user_id:
                try:
                    actual_duration = get_media_duration_seconds(out_path)
                    cost = calculate_generation_cost(actual_duration)
                    charge_user(db, owner_user_id, cost)
                except HTTPException:
                    raise
                except Exception as e:
                    # Не блокируем выдачу готового видео из-за сбоя биллинга —
                    # просто логируем, разберёмся постфактум.
                    print(f"Не удалось списать средства за задачу {job_id}: {e}")

        return FileResponse(out_path, media_type="video/mp4", filename="avatar_result.mp4")

    if status == "FAILED":
        raise HTTPException(status_code=502, detail=f"Генерация упала: {data}")

    return {"job_id": job_id, "status": status}


@router.get("/debug/runpod-ping")
async def debug_runpod_ping():
    """Временный диагностический эндпоинт — см. историю чата. Можно
    удалить, когда стабильность будет подтверждена окончательно."""
    endpoints = {"sadtalker_xtts": RUNPOD_BASE_URL, "lipsync": RUNPOD_LIPSYNC_BASE_URL}
    results = {}
    for name, base_url in endpoints.items():
        start = time.time()
        try:
            resp = await asyncio.to_thread(requests.get, f"{base_url}/health", headers=HEADERS, timeout=20)
            elapsed = round(time.time() - start, 2)
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:500]
            results[name] = {"ok": True, "status_code": resp.status_code, "elapsed_sec": elapsed, "body": body}
        except requests.exceptions.RequestException as e:
            results[name] = {"ok": False, "elapsed_sec": round(time.time() - start, 2), "error": str(e)}
    results["env_check"] = {
        "runpod_api_key_set": bool(RUNPOD_API_KEY),
        "runpod_endpoint_id_set": bool(RUNPOD_ENDPOINT_ID),
        "runpod_lipsync_endpoint_id_set": bool(RUNPOD_LIPSYNC_ENDPOINT_ID),
    }
    return results


@router.get("/debug/runpod-run-ping")
async def debug_runpod_run_ping():
    """Проверяет именно POST /run — см. историю чата."""
    endpoints = {"sadtalker_xtts": RUNPOD_BASE_URL, "lipsync": RUNPOD_LIPSYNC_BASE_URL}
    results = {}
    for name, base_url in endpoints.items():
        entry = {}
        start = time.time()
        try:
            resp = await asyncio.to_thread(
                requests.post, f"{base_url}/run", headers=HEADERS,
                json={"input": {"healthcheck": True}}, timeout=30
            )
            entry["run_elapsed_sec"] = round(time.time() - start, 2)
            resp.raise_for_status()
            job = resp.json()
            entry["run_status_code"] = resp.status_code
            entry["run_response"] = job
            job_id = job.get("id")
            if job_id:
                start2 = time.time()
                status_resp = await asyncio.to_thread(
                    requests.get, f"{base_url}/status/{job_id}", headers=HEADERS, timeout=20
                )
                entry["status_elapsed_sec"] = round(time.time() - start2, 2)
                entry["status_response"] = status_resp.json()
        except requests.exceptions.RequestException as e:
            entry["error"] = str(e)
            entry["elapsed_sec_before_error"] = round(time.time() - start, 2)
        results[name] = entry
    return results
