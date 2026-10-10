import os
import json
import base64
import time
import uuid
import shutil
import asyncio
import requests
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from mutagen import File as MutagenFile
from dotenv import load_dotenv

from database import get_db
from auth_utils import get_current_user_id
from routers.billing import calculate_generation_cost, get_or_create_user, charge_user as charge_feature

load_dotenv()

RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

RUNPOD_LIPSYNC_ENDPOINT_ID = os.getenv("RUNPOD_LIPSYNC_ENDPOINT_ID")
RUNPOD_LIPSYNC_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_LIPSYNC_ENDPOINT_ID}"

# Pro-тир Кейсов 1 и 2 — отдельный RunPod-эндпоинт на EchoMimicV2
# (репозиторий Avatar-Studio-EchoMimic). Не путать с RUNPOD_ENDPOINT_ID
# (SadTalker/XTTS, Basic-тир) и RUNPOD_LIPSYNC_ENDPOINT_ID (MuseTalk, Кейс 3).
RUNPOD_ECHOMIMIC_ENDPOINT_ID = os.getenv("RUNPOD_ECHOMIMIC_ENDPOINT_ID")
RUNPOD_ECHOMIMIC_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ECHOMIMIC_ENDPOINT_ID}"

# Pro-тир Кейсов 1 и 2 — ЗАМЕНА EchoMimicV2 на LongCat-Video-Avatar 1.5
# после того, как EchoMimicV2 был признан неприемлемым по качеству для
# коммерческого продукта (сильные искажения лица/пропорций даже на
# самом спокойном pose-шаблоне, не чинится дальнейшей настройкой —
# см. историю чата). Инфраструктура EchoMimic (переменные и функция
# выше/ниже) оставлена в коде нетронутой на случай пересмотра, но
# больше не используется ни в одном активном маршруте.
RUNPOD_LONGCAT_ENDPOINT_ID = os.getenv("RUNPOD_LONGCAT_ENDPOINT_ID")
RUNPOD_LONGCAT_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_LONGCAT_ENDPOINT_ID}"

UNLIMITED_USER_IDS = {u.strip() for u in os.getenv("UNLIMITED_USER_IDS", "").split(",") if u.strip()}

TEMP_DIR = "data/runpod_tmp"
OUTPUT_DIR = "data/outputs"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_CLIP_SECONDS = 15  # то же ограничение, что и во фронтенде (api.js)

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

# Pro-тир Кейса 1 (фото + текст + голос + эмоция, EchoMimicV2): та же
# идея, что и у Кейса 3 text-mode выше — TTS должен отработать ДО
# отправки в EchoMimic, поэтому это тоже фоновая задача с временным
# job_id, пока не превратится в реальный RunPod job_id с префиксом
# echomimic: (см. _run_case1_pro_job и get_job_status).
ECHOMIMIC_PRO_TEXT_PREFIX = "echomimicpro:"
ECHOMIMIC_PRO_TEXT_JOBS: dict[str, dict] = {}

# Pro-тир Кейса 2 (фото + готовое аудио, EchoMimicV2) отправляется в
# EchoMimic-воркер напрямую, без промежуточного TTS-шага — поэтому
# отдельный фоновый JOBS-словарь ему не нужен, только префикс для
# маршрутизации опроса статуса на правильный RunPod-эндпоинт.
ECHOMIMIC_PREFIX = "echomimic:"

# Активный Pro-движок (см. комментарий у RUNPOD_LONGCAT_ENDPOINT_ID выше).
LONGCAT_PREFIX = "longcat:"
# Эмоции, доступные на форме Pro (ключи совпадают с библиотекой в handler.py LongCat).
PRO_EMOTIONS = {"neutral", "angry", "happy", "sad"}

JOB_OWNERS: dict[str, str] = {}
# Цена, озвученная клиенту при отправке (Кейс 2): списываем именно её, когда
# видео готово. Ключ — тот же job_id, что и в JOB_OWNERS.
JOB_COSTS: dict[str, float] = {}
# Когда впервые увидели задачу в статусе IN_QUEUE (для отмены долгого ожидания GPU).
JOB_FIRST_SEEN: dict[str, float] = {}

# Если задача ждёт свободную GPU дольше этого времени — отменяем её (деньги
# не списываются, клиент видит понятное сообщение). Настраивается на Railway.
QUEUE_WAIT_LIMIT_SECONDS = int(os.getenv("QUEUE_WAIT_LIMIT_SECONDS", "600"))
# Общий срок жизни LongCat-задачи в RunPod (ожидание + выполнение), секунд.
# Страховка для брошенных задач (клиент закрыл вкладку): без неё задача висит
# 24 часа и может запуститься, когда она уже никому не нужна, — а GPU платная.
# ВАЖНО: это жёсткий лимит, он должен покрывать и ожидание, и саму генерацию.
LONGCAT_JOB_TTL_SECONDS = int(os.getenv("LONGCAT_JOB_TTL_SECONDS", "3600"))

# ---------- Озвучка текста для Video Studio ----------
# Цена для клиента за 1 секунду готовой озвучки. ЭТО СТАРТОВОЕ ЗНАЧЕНИЕ:
# пересчитайте по факту замера в RunPod (как у RUNPOD_COST_PER_SECOND_USD в
# billing.py) и поменяйте переменной TTS_PRICE_PER_SECOND_USD на Railway.
TTS_PRICE_PER_SECOND_USD = float(os.getenv("TTS_PRICE_PER_SECOND_USD", "0.006"))
TTS_MAX_CHARS = 1500
TTS_JOB_TTL_SECONDS = int(os.getenv("TTS_JOB_TTL_SECONDS", "900"))
# Ключи — то, что шлёт Video Studio; значения — имена встроенных голосов XTTS-v2.
# Проверьте имена запросом {"input": {"list_speakers": true}} к воркеру
# SadTalker/XTTS и при необходимости поправьте здесь.
TTS_VOICES = {
    "f1": "Claribel Dervla",
    "f2": "Daisy Studious",
    "m1": "Andrew Chipper",
    "m2": "Damien Black",
}
TTS_VOICE_STYLES = {"neutral", "lower", "higher", "warm", "bright"}
TTS_TEMPOS = {"slow", "normal", "fast"}
TTS_JOBS: dict[str, dict] = {}


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


def ensure_balance(db: Session, user_id: str, cost: float):
    """Проверяет, что на балансе хватает денег, НО ничего не списывает.
    Списание происходит позже — когда видео готово (см. get_job_status)."""
    if user_id in UNLIMITED_USER_IDS:
        return
    user = get_or_create_user(db, user_id)
    if user.balance_usd < cost:
        raise HTTPException(
            status_code=402,
            detail=f"Недостаточно средств: нужно ${cost:.4f}, на балансе ${user.balance_usd:.4f}"
        )


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


def submit_echomimic_job(image_path: str, audio_path: str, emotion: Optional[str] = None) -> str:
    """Pro-тир Кейсов 1 и 2 — EchoMimicV2-воркер. emotion=None означает
    'определи эмоцию сам по аудио' (Кейс 2 Pro, см. handler.py в
    Avatar-Studio-EchoMimic); для Кейса 1 Pro emotion всегда передаётся
    явно (пользователь выбрал её в UI)."""
    if not RUNPOD_ECHOMIMIC_ENDPOINT_ID:
        raise HTTPException(
            status_code=500,
            detail="RUNPOD_ECHOMIMIC_ENDPOINT_ID не задан на сервере — проверь переменные окружения Railway"
        )

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload_input = {
        "image_base64": img_b64,
        "audio_base64": audio_b64,
        "max_seconds": MAX_CLIP_SECONDS,
    }
    if emotion:
        payload_input["emotion"] = emotion

    try:
        resp = requests.post(f"{RUNPOD_ECHOMIMIC_BASE_URL}/run", headers=HEADERS, json={"input": payload_input}, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Не удалось связаться с RunPod (EchoMimic): {e}")

    job = resp.json()
    raw_job_id = job.get("id")
    if not raw_job_id:
        raise HTTPException(status_code=502, detail=f"RunPod (EchoMimic) не вернул id задачи: {job}")

    return f"{ECHOMIMIC_PREFIX}{raw_job_id}"


def submit_longcat_job(image_path: str, audio_path: str, emotion: Optional[str] = None) -> str:
    """Активный Pro-тир Кейсов 1 и 2 — LongCat-Video-Avatar 1.5
    (см. Avatar-Studio-LongCat). emotion передаётся как есть (или None
    для нейтрального дефолта) — воркер сам превращает её в текстовый
    промпт, стиля pose-библиотек, как у EchoMimic, здесь нет."""
    if not RUNPOD_LONGCAT_ENDPOINT_ID:
        raise HTTPException(
            status_code=500,
            detail="RUNPOD_LONGCAT_ENDPOINT_ID не задан на сервере — проверь переменные окружения Railway"
        )

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload_input = {
        "image_base64": img_b64,
        "audio_base64": audio_b64,
    }
    if emotion:
        payload_input["emotion"] = emotion

    try:
        resp = requests.post(
            f"{RUNPOD_LONGCAT_BASE_URL}/run", headers=HEADERS,
            json={"input": payload_input, "policy": {"ttl": LONGCAT_JOB_TTL_SECONDS * 1000}},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Не удалось связаться с RunPod (LongCat): {e}")

    job = resp.json()
    raw_job_id = job.get("id")
    if not raw_job_id:
        raise HTTPException(status_code=502, detail=f"RunPod (LongCat) не вернул id задачи: {job}")

    return f"{LONGCAT_PREFIX}{raw_job_id}"


def poll_runpod_job_sync(base_url: str, job_id: str, timeout: float = 120, interval: float = 3) -> dict:
    """Синхронно ждёт завершения RunPod-задачи (TTS) — вызывается из
    _run_lipsync_from_text_job/_run_case1_pro_job через asyncio.to_thread,
    т.е. уже вне event loop, так что этот таймаут больше не рискует
    упереться ни в клиентский, ни в Railway-прокси таймаут — сам
    HTTP-запрос давно закончился к этому моменту."""
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


async def _run_case1_pro_job(internal_id: str, image_path: str, voice_path: str,
                              work_dir: str, text: str, emotion: str, language: str, user_id: str):
    """Фоновая обработка Кейса 1 Pro — сначала озвучка текста через
    существующий SadTalker/XTTS-воркер (tts_only, та же функция
    synthesize_tts_via_runpod, что и у Кейса 3 text-mode — TTS-логику
    не дублируем), затем готовое аудио вместе с фото и явно выбранной
    пользователем эмоцией уходит в EchoMimic-воркер. Тот же паттерн
    фоновой задачи, что и у _run_lipsync_from_text_job выше."""
    try:
        audio_path, duration = await asyncio.to_thread(
            synthesize_tts_via_runpod, voice_path, text, language, work_dir
        )
        runpod_job_id = await asyncio.to_thread(submit_longcat_job, image_path, audio_path, emotion)
        JOB_OWNERS[runpod_job_id] = user_id
        ECHOMIMIC_PRO_TEXT_JOBS[internal_id] = {"status": "SUBMITTED", "runpod_job_id": runpod_job_id, "error": None}
    except HTTPException as e:
        ECHOMIMIC_PRO_TEXT_JOBS[internal_id] = {"status": "FAILED", "runpod_job_id": None, "error": str(e.detail)}
    except Exception as e:
        ECHOMIMIC_PRO_TEXT_JOBS[internal_id] = {"status": "FAILED", "runpod_job_id": None, "error": str(e)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@router.post("/photo-emotion")
async def generate_photo_emotion(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    expression_scale: float = Form(0.7),
    pose_style: int = Form(0),
    tier: str = Form("basic"),
    emotion: str = Form("neutral"),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Кейс 2. tier="pro" отправляет задачу в LongCat-Video-Avatar.
    Эмоцию (neutral / angry / happy / sad) пользователь выбирает на
    форме перед генерацией — воркер превращает её в текстовый промпт.
    expression_scale/pose_style в Pro-режиме не используются (это
    параметры SadTalker), но по-прежнему принимаются с фронтенда без
    ошибки, чтобы не усложнять форму."""
    request_id = uuid.uuid4().hex
    image_path = os.path.join(TEMP_DIR, f"{request_id}_{image.filename}")
    audio_path = os.path.join(TEMP_DIR, f"{request_id}_{audio.filename}")

    with open(image_path, "wb") as f:
        f.write(await image.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    try:
        duration = get_media_duration_seconds(audio_path)
        cost = calculate_generation_cost(duration, tier)
        # Защита средств клиента: заранее ТОЛЬКО проверяем баланс. Списываем
        # после того, как видео готово и отдано (get_job_status) — при ошибке,
        # OOM или отмене из-за нехватки GPU деньги остаются у клиента.
        ensure_balance(db, user_id, cost)

        if tier == "pro":
            pro_emotion = emotion if emotion in PRO_EMOTIONS else "neutral"
            job_id = await asyncio.to_thread(submit_longcat_job, image_path, audio_path, pro_emotion)
        else:
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

    JOB_OWNERS[job_id] = user_id
    JOB_COSTS[job_id] = cost
    return {"job_id": job_id, "status_url": f"/api/generate/status/{job_id}"}


@router.post("/photo-text-emotion")
async def generate_photo_text_emotion(
    image: UploadFile = File(...),
    voice_sample: UploadFile = File(...),
    text: str = Form(...),
    emotion: str = Form("neutral"),
    language: str = Form("ru"),
    tier: str = Form("basic"),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Кейс 1. tier="basic" — прежнее поведение без изменений (один
    синхронный вызов SadTalker/XTTS-воркера, который делает TTS и
    видео вместе). tier="pro" — фоновая задача: сначала TTS через тот
    же SadTalker/XTTS-воркер (tts_only), затем EchoMimic с явно
    выбранной эмоцией; списание баланса в Pro-режиме отложено до
    готового видео (тот же механизм JOB_OWNERS, что и у Кейса 3
    text-mode), т.к. точная стоимость известна только после TTS."""
    if tier == "pro":
        # Минимальный порог: на балансе должно хватать хотя бы на один
        # сегмент (~3.7 с) Pro-видео. Точная сумма — по готовому видео.
        ensure_balance(db, user_id, calculate_generation_cost(3.72, "pro"))
        request_id = uuid.uuid4().hex
        work_dir = os.path.join(TEMP_DIR, request_id)
        os.makedirs(work_dir, exist_ok=True)
        image_path = os.path.join(work_dir, f"image_{image.filename}")
        voice_path = os.path.join(work_dir, f"voice_{voice_sample.filename}")

        with open(image_path, "wb") as f:
            f.write(await image.read())
        with open(voice_path, "wb") as f:
            f.write(await voice_sample.read())

        internal_id = uuid.uuid4().hex
        ECHOMIMIC_PRO_TEXT_JOBS[internal_id] = {"status": "PENDING", "runpod_job_id": None, "error": None}

        asyncio.create_task(
            _run_case1_pro_job(internal_id, image_path, voice_path, work_dir, text, emotion, language, user_id)
        )

        job_id = f"{ECHOMIMIC_PRO_TEXT_PREFIX}{internal_id}"
        return {"job_id": job_id, "status_url": f"/api/generate/status/{job_id}"}

    # ------- tier="basic": прежнее поведение без изменений -------
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


def submit_tts_job(text: str, language: str, voice_style: str, speech_tempo: str,
                   voice_sample_path: Optional[str], speaker: Optional[str]) -> str:
    """Озвучка текста на воркере SadTalker/XTTS (режим tts_only)."""
    inp = {
        "tts_only": True,
        "text": text,
        "language": language,
        "voice_style": voice_style,
        "speech_tempo": speech_tempo,
    }
    if voice_sample_path:
        with open(voice_sample_path, "rb") as f:
            inp["voice_sample_base64"] = base64.b64encode(f.read()).decode("utf-8")
    else:
        inp["speaker"] = speaker

    try:
        resp = requests.post(
            f"{RUNPOD_BASE_URL}/run", headers=HEADERS,
            json={"input": inp, "policy": {"ttl": TTS_JOB_TTL_SECONDS * 1000}},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Не удалось связаться с RunPod (озвучка): {e}")

    job_id = resp.json().get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail="RunPod не вернул id задачи озвучки")
    return job_id


@router.post("/tts")
async def generate_tts(
    text: str = Form(...),
    language: str = Form("ru"),
    speaker: str = Form(""),
    voice_style: str = Form("neutral"),
    speech_tempo: str = Form("normal"),
    voice_sample: Optional[UploadFile] = File(None),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Озвучка текста для Video Studio. Встроенный голос (speaker = f1/f2/m1/m2)
    или клонирование по образцу (voice_sample). Деньги списываются только
    когда готовый звук отдан клиенту (см. /tts-status)."""
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Введите текст для озвучки")
    if len(text) > TTS_MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"Слишком длинный текст: максимум {TTS_MAX_CHARS} символов")
    if voice_style not in TTS_VOICE_STYLES or speech_tempo not in TTS_TEMPOS:
        raise HTTPException(status_code=400, detail="Неизвестный тембр или темп речи")

    # Предварительная проверка баланса по длине текста (≈13 символов в секунду
    # речи); берём половину оценки, точная сумма считается по готовому звуку.
    est_cost = round(max(2.0, len(text) / 13.0) * TTS_PRICE_PER_SECOND_USD, 4)
    ensure_balance(db, user_id, round(est_cost * 0.5, 4))

    sample_path = None
    if voice_sample is not None and voice_sample.filename:
        data = await voice_sample.read()
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Образец голоса слишком большой")
        sample_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_voice.wav")
        with open(sample_path, "wb") as f:
            f.write(data)
    elif speaker not in TTS_VOICES:
        raise HTTPException(status_code=400, detail="Выберите голос")

    try:
        job_id = await asyncio.to_thread(
            submit_tts_job, text, language, voice_style, speech_tempo,
            sample_path, TTS_VOICES.get(speaker),
        )
    finally:
        if sample_path and os.path.exists(sample_path):
            os.remove(sample_path)

    TTS_JOBS[job_id] = {"user_id": user_id, "chars": len(text), "paid": False, "lock": asyncio.Lock()}
    return {"job_id": job_id, "status_url": f"/api/generate/tts-status/{job_id}", "estimate_usd": est_cost}


@router.get("/tts-status/{job_id}")
async def get_tts_status(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Опрос озвучки. Пока идёт — JSON со статусом; когда готово — WAV-файл.
    Списание происходит один раз, при первой выдаче готового файла; если на
    балансе не хватает (402), файл не отдаётся и можно повторить опрос после
    пополнения."""
    entry = TTS_JOBS.get(job_id)
    if entry is None or entry["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Задача не найдена (возможно, сервер перезапускался)")

    out_path = os.path.join(OUTPUT_DIR, f"tts_{job_id}.wav")
    if entry["paid"] and os.path.exists(out_path):
        return FileResponse(out_path, media_type="audio/wav", filename="voiceover.wav")

    resp = await asyncio.to_thread(requests.get, f"{RUNPOD_BASE_URL}/status/{job_id}", headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        TTS_JOBS.pop(job_id, None)
        raise HTTPException(status_code=504, detail="Время ожидания озвучки истекло. Деньги не списаны, попробуйте ещё раз.")

    data = resp.json()
    status = data.get("status")

    if status == "COMPLETED":
        output = data.get("output") or {}
        if output.get("error"):
            TTS_JOBS.pop(job_id, None)
            raise HTTPException(status_code=502, detail=f"Озвучка не удалась: {output['error']}. Деньги не списаны.")
        audio_b64 = output.get("audio_base64")
        if not audio_b64:
            TTS_JOBS.pop(job_id, None)
            raise HTTPException(status_code=502, detail="Озвучка не вернула звук. Деньги не списаны.")

        async with entry["lock"]:
            if not entry["paid"]:
                def _write():
                    with open(out_path, "wb") as f:
                        f.write(base64.b64decode(audio_b64))

                await asyncio.to_thread(_write)
                try:
                    duration = get_media_duration_seconds(out_path)
                    if user_id not in UNLIMITED_USER_IDS:
                        cost = round(max(duration, 1.0) * TTS_PRICE_PER_SECOND_USD, 4)
                        charge_feature(
                            db, user_id, cost, feature="tts",
                            units=round(duration, 2), note=f"{entry['chars']} симв.",
                        )
                    entry["paid"] = True
                except Exception:
                    if os.path.exists(out_path):
                        os.remove(out_path)
                    raise
        return FileResponse(out_path, media_type="audio/wav", filename="voiceover.wav")

    if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
        TTS_JOBS.pop(job_id, None)
        raise HTTPException(status_code=502, detail=f"Озвучка не удалась ({status}). Деньги не списаны.")

    return {"job_id": job_id, "status": status}


@router.get("/status/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Опрашивается клиентом до тех пор, пока генерация не завершится.

    job_id с префиксом "textlipsync:" или "echomimicpro:" — фоновые
    задачи (TTS ещё не завершился, реального RunPod job_id пока нет,
    см. _run_lipsync_from_text_job / _run_case1_pro_job); "lipsync:" —
    MuseTalk (Кейс 3 basic); "echomimic:" — EchoMimicV2 (Pro-тир
    Кейсов 1 и 2); всё остальное — основной эндпоинт (SadTalker/XTTS,
    Кейсы 1 и 2 Basic)."""
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

    if job_id.startswith(ECHOMIMIC_PRO_TEXT_PREFIX):
        internal_id = job_id[len(ECHOMIMIC_PRO_TEXT_PREFIX):]
        entry = ECHOMIMIC_PRO_TEXT_JOBS.get(internal_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Задача не найдена (возможно, сервер перезапускался)")
        if entry["status"] == "PENDING":
            return {"job_id": job_id, "status": "PENDING"}
        if entry["status"] == "FAILED":
            raise HTTPException(status_code=502, detail=f"Генерация упала: {entry['error']}")
        job_id = entry["runpod_job_id"]

    if job_id.startswith(LIPSYNC_PREFIX):
        raw_job_id = job_id[len(LIPSYNC_PREFIX):]
        base_url = RUNPOD_LIPSYNC_BASE_URL
    elif job_id.startswith(ECHOMIMIC_PREFIX):
        raw_job_id = job_id[len(ECHOMIMIC_PREFIX):]
        base_url = RUNPOD_ECHOMIMIC_BASE_URL
    elif job_id.startswith(LONGCAT_PREFIX):
        raw_job_id = job_id[len(LONGCAT_PREFIX):]
        base_url = RUNPOD_LONGCAT_BASE_URL
    else:
        raw_job_id = job_id
        base_url = RUNPOD_BASE_URL

    status_url = f"{base_url}/status/{raw_job_id}"
    status_resp = await asyncio.to_thread(requests.get, status_url, headers=HEADERS, timeout=30)

    def _forget_job():
        JOB_OWNERS.pop(job_id, None)
        JOB_COSTS.pop(job_id, None)
        JOB_FIRST_SEEN.pop(job_id, None)

    # RunPod удаляет задачу по TTL — тогда статус отвечает 404.
    if status_resp.status_code == 404:
        _forget_job()
        raise HTTPException(
            status_code=504,
            detail="Время ожидания задачи истекло. Деньги за неё не списаны — попробуйте ещё раз."
        )

    data = status_resp.json()
    status = data.get("status")

    # Долго ждём свободную GPU — отменяем задачу, деньги не списываются.
    if status == "IN_QUEUE":
        waited = time.time() - JOB_FIRST_SEEN.setdefault(job_id, time.time())
        if waited > QUEUE_WAIT_LIMIT_SECONDS:
            try:
                await asyncio.to_thread(requests.post, f"{base_url}/cancel/{raw_job_id}", headers=HEADERS, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"Не удалось отменить задачу {job_id}: {e}")
            _forget_job()
            raise HTTPException(
                status_code=503,
                detail="Сейчас нет свободных видеокарт. Задача отменена, деньги не списаны — попробуйте через несколько минут."
            )

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
            quoted_cost = JOB_COSTS.pop(job_id, None)
            JOB_FIRST_SEEN.pop(job_id, None)
            if owner_user_id:
                try:
                    if quoted_cost is not None:
                        # Кейс 2: списываем ровно ту сумму, что показали клиенту.
                        cost = quoted_cost
                    else:
                        actual_duration = get_media_duration_seconds(out_path)
                        charge_tier = "pro" if job_id.startswith(LONGCAT_PREFIX) else "basic"
                        cost = calculate_generation_cost(actual_duration, charge_tier)
                    charge_user(db, owner_user_id, cost)
                except HTTPException:
                    raise
                except Exception as e:
                    print(f"Не удалось списать средства за задачу {job_id}: {e}")

        return FileResponse(out_path, media_type="video/mp4", filename="avatar_result.mp4")

    if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
        _forget_job()
        raise HTTPException(status_code=502, detail=f"Генерация упала: {data}. Деньги за неё не списаны.")

    return {"job_id": job_id, "status": status}
