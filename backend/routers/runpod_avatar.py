import os
import json
import base64
import time
import uuid
import asyncio
import requests
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()

RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# Кейс 3 (MuseTalk-липсинк) живёт на отдельном RunPod Serverless
# эндпоинте (Avatar-Studio-Lipsync), отдельном от SadTalker/XTTS
# (Avatar-Studio). API-ключ общий, endpoint ID — свой.
RUNPOD_LIPSYNC_ENDPOINT_ID = os.getenv("RUNPOD_LIPSYNC_ENDPOINT_ID")
RUNPOD_LIPSYNC_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_LIPSYNC_ENDPOINT_ID}"

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
# чтобы GET /status/{job_id} знал, к какому RunPod-воркеру стучаться.
# Формат отдаваемого клиенту job_id: "lipsync:<runpod_job_id>".
LIPSYNC_PREFIX = "lipsync:"


# ---------------------------------------------------------------------------
# Важно: эндпоинты FastAPI ниже НЕ ждут завершения генерации внутри одного
# HTTP-запроса. Railway (edge-прокси перед приложением) обрывает долгие
# запросы своим собственным таймаутом независимо от кода приложения —
# генерация видео (SadTalker + XTTS-v2 / MuseTalk) может занимать 5+ минут,
# что превышает этот лимит. Поэтому используется схема submit → poll:
#   1) POST /photo-text-emotion (или /photo-emotion, /lipsync) сразу
#      отправляет задачу в RunPod и возвращает job_id — сам HTTP-запрос
#      занимает секунды.
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

    # ВАЖНО: запрос к RunPod оборачиваем в try/except. Раньше сетевая
    # ошибка (недоступный эндпоинт, неверный ID, таймаут и т.п.) здесь
    # улетала наверх необработанной, что на стороне Railway иногда
    # приводит к обрыву соединения без внятного ответа — на фронтенде
    # это выглядит как generic "Failed to fetch" без единой зацепки,
    # что именно сломалось.
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
    """Кейс 3: отдельный RunPod-эндпоинт (MuseTalk). Возвращаемый job_id
    помечается префиксом LIPSYNC_PREFIX, чтобы /status/{job_id} знал,
    к какому воркеру идти за результатом."""
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
        # ВАЖНО: submit_sadtalker_job внутри делает синхронный (блокирующий)
        # requests.post к RunPod. Вызванный напрямую внутри async-эндпоинта,
        # такой блокирующий вызов останавливает ВЕСЬ event loop на время
        # ожидания ответа от RunPod — сервер перестаёт отвечать вообще всем
        # клиентам, что выглядит как массовый "Failed to fetch". Поэтому
        # выполняем его в отдельном потоке через asyncio.to_thread.
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
        # См. комментарий в generate_photo_emotion — уводим блокирующий
        # HTTP-вызов к RunPod в отдельный поток, чтобы не вешать сервер.
        job_id = await asyncio.to_thread(
            submit_photo_text_emotion_job,
            image_path, voice_path, text, emotion, language
        )
    finally:
        for p in (image_path, voice_path):
            if os.path.exists(p):
                os.remove(p)

    return {"job_id": job_id, "status_url": f"/api/generate/status/{job_id}"}


@router.post("/lipsync")
async def generate_lipsync(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    """Кейс 3: видео с лицом + аудио-драйвер → липсинк через MuseTalk 1.5
    на отдельном RunPod-эндпоинте. Сразу возвращает job_id — результат
    забирается через GET /status/{job_id}, как и для кейсов 1/2."""
    request_id = uuid.uuid4().hex
    video_path = os.path.join(TEMP_DIR, f"{request_id}_{video.filename}")
    audio_path = os.path.join(TEMP_DIR, f"{request_id}_{audio.filename}")

    with open(video_path, "wb") as f:
        f.write(await video.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    try:
        # См. комментарий в generate_photo_emotion — уводим блокирующий
        # HTTP-вызов к RunPod в отдельный поток, чтобы не вешать сервер.
        job_id = await asyncio.to_thread(submit_lipsync_job, video_path, audio_path)
    finally:
        for p in (video_path, audio_path):
            if os.path.exists(p):
                os.remove(p)

    return {"job_id": job_id, "status_url": f"/api/generate/status/{job_id}"}


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Опрашивается клиентом (или вручную через Swagger/curl) до тех пор,
    пока генерация не завершится. Быстрый запрос — не подвержен таймауту
    Railway edge-прокси, в отличие от прямого ожидания результата.

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
    # Тот же блокирующий HTTP-вызов, что и при отправке задачи — этот
    # эндпоинт дёргается фронтендом каждые несколько секунд, поэтому
    # особенно важно не блокировать event loop именно здесь.
    status_resp = await asyncio.to_thread(requests.get, status_url, headers=HEADERS, timeout=30)
    data = status_resp.json()
    status = data.get("status")

    if status == "COMPLETED":
        video_b64 = data["output"]["video_base64"]
        out_path = os.path.join(OUTPUT_DIR, f"{raw_job_id}.mp4")

        def _write_video():
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(video_b64))

        # Декодирование base64 и запись на диск — тоже блокирующие
        # операции, для крупных видео могут занимать заметное время.
        await asyncio.to_thread(_write_video)
        return FileResponse(out_path, media_type="video/mp4", filename="avatar_result.mp4")

    if status == "FAILED":
        raise HTTPException(status_code=502, detail=f"Генерация упала: {data}")

    # IN_QUEUE / IN_PROGRESS и т.п. — сообщаем клиенту, что нужно спросить позже
    return {"job_id": job_id, "status": status}


@router.get("/debug/runpod-ping")
async def debug_runpod_ping():
    """Временный диагностический эндпоинт: проверяет сетевую связность
    Railway -> RunPod напрямую (через /health каждого эндпоинта, без
    траты GPU-времени) и замеряет, сколько это реально занимает.
    Открывается просто как ссылка в браузере — не требует Swagger,
    curl или ручного ввода токенов. Удалить после того, как проблема
    с "Failed to fetch" будет найдена и решена."""
    endpoints = {
        "sadtalker_xtts": RUNPOD_BASE_URL,
        "lipsync": RUNPOD_LIPSYNC_BASE_URL,
    }
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
            elapsed = round(time.time() - start, 2)
            results[name] = {"ok": False, "elapsed_sec": elapsed, "error": str(e)}

    results["env_check"] = {
        "runpod_api_key_set": bool(RUNPOD_API_KEY),
        "runpod_endpoint_id_set": bool(RUNPOD_ENDPOINT_ID),
        "runpod_lipsync_endpoint_id_set": bool(RUNPOD_LIPSYNC_ENDPOINT_ID),
    }
    return results


@router.get("/debug/runpod-run-ping")
async def debug_runpod_run_ping():
    """Проверяет именно POST /run (постановку задачи в очередь) — а не
    /health, который лишь подтверждает, что воркер жив, но не проходит
    через ту же логику приёма задачи. Использует поддерживаемый обоими
    воркерами режим {"input": {"healthcheck": true}} — он завершается
    почти мгновенно на стороне воркера, без реальной GPU-генерации, так
    что тест ничего не стоит по деньгам и времени GPU."""
    endpoints = {
        "sadtalker_xtts": RUNPOD_BASE_URL,
        "lipsync": RUNPOD_LIPSYNC_BASE_URL,
    }
    results = {}
    for name, base_url in endpoints.items():
        entry = {}
        # Шаг 1: POST /run с healthcheck-пейлоадом
        start = time.time()
        try:
            resp = await asyncio.to_thread(
                requests.post, f"{base_url}/run", headers=HEADERS,
                json={"input": {"healthcheck": True}}, timeout=30
            )
            run_elapsed = round(time.time() - start, 2)
            resp.raise_for_status()
            job = resp.json()
            entry["run_elapsed_sec"] = run_elapsed
            entry["run_status_code"] = resp.status_code
            entry["run_response"] = job
            job_id = job.get("id")

            # Шаг 2: сразу опрашиваем статус — healthcheck-задача должна
            # завершиться почти мгновенно
            if job_id:
                start2 = time.time()
                status_resp = await asyncio.to_thread(
                    requests.get, f"{base_url}/status/{job_id}", headers=HEADERS, timeout=20
                )
                status_elapsed = round(time.time() - start2, 2)
                entry["status_elapsed_sec"] = status_elapsed
                entry["status_response"] = status_resp.json()
        except requests.exceptions.RequestException as e:
            entry["error"] = str(e)
            entry["elapsed_sec_before_error"] = round(time.time() - start, 2)

        results[name] = entry

    return results
