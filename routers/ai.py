"""
Avatar Studio — ИИ-шлюз для Video Studio: расшифровка речи через OpenRouter.

Ключ OpenRouter хранится только здесь (переменная окружения), в браузер он
не уходит. Оплата — с общего баланса (billing.charge_user), после успешной
расшифровки. Вход проверяется через Supabase, как у остальных эндпоинтов.

Переменные окружения:
  OPENROUTER_API_KEY        — ключ OpenRouter (уже задан на Railway)
  STT_MODEL                 — модель распознавания (по умолчанию openai/whisper-large-v3-turbo)
  STT_PRICE_PER_SECOND_USD  — цена для клиента за секунду аудио (по умолчанию 0.00005, ≈ $0.18 за час)
  UNLIMITED_USER_IDS        — id пользователей без списаний (как в runpod_avatar.py)
"""
import os
import uuid
import asyncio
import requests
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from mutagen import File as MutagenFile
from dotenv import load_dotenv

from database import get_db
from auth_utils import get_current_user_id
from routers.billing import get_or_create_user, charge_user as charge_feature

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_STT_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
STT_MODEL = os.getenv("STT_MODEL", "openai/whisper-large-v3-turbo")
# Цена для клиента за 1 секунду аудио. Стартовое значение: пересчитайте по факту
# списаний на openrouter.ai/activity и поменяйте переменной на Railway.
STT_PRICE_PER_SECOND_USD = float(os.getenv("STT_PRICE_PER_SECOND_USD", "0.00005"))
STT_MIN_BILLABLE_SECONDS = 5.0
STT_MAX_BYTES = 24 * 1024 * 1024
STT_MAX_SECONDS = 60 * 60
UNLIMITED_USER_IDS = {u.strip() for u in os.getenv("UNLIMITED_USER_IDS", "").split(",") if u.strip()}

TEMP_DIR = "data/runpod_tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

router = APIRouter(prefix="/api/ai", tags=["ai"])

_stt_sem: Optional[asyncio.Semaphore] = None


def _sem() -> asyncio.Semaphore:
    """Не больше 4 расшифровок одновременно, чтобы не забить сервер."""
    global _stt_sem
    if _stt_sem is None:
        _stt_sem = asyncio.Semaphore(4)
    return _stt_sem


def _audio_duration(path: str) -> Optional[float]:
    try:
        media = MutagenFile(path)
        if media is not None and media.info is not None and hasattr(media.info, "length"):
            return float(media.info.length)
    except Exception:
        pass
    return None


def _error_text(resp) -> str:
    try:
        err = resp.json().get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
    except Exception:
        pass
    return f"код {resp.status_code}"


def _call_openrouter_stt(path: str, language: Optional[str]) -> dict:
    """Синхронный вызов OpenRouter Speech-to-Text (multipart, формат OpenAI).
    Вызывается только через asyncio.to_thread — иначе заблокирует event loop."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "X-Title": "BestConsulting Video Studio",
    }
    data = [
        ("model", STT_MODEL),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
    ]
    if language:
        data.append(("language", language))
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                OPENROUTER_STT_URL, headers=headers, data=data,
                files={"file": ("audio.mp3", f, "audio/mpeg")}, timeout=180,
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Не удалось связаться с сервисом расшифровки: {e}")

    if resp.status_code == 402:
        # на нашем счёте OpenRouter закончились кредиты: клиенту подробности не нужны
        print(f"OpenRouter STT: 402 Payment Required — пополните кредиты OpenRouter. {_error_text(resp)}")
        raise HTTPException(status_code=503, detail="Расшифровка временно недоступна. Попробуйте позже.")
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Расшифровка не удалась: {_error_text(resp)}")
    return resp.json()


def _normalize(data: dict, duration: Optional[float]) -> dict:
    """Приводит ответ провайдера к виду {words: [{w, s, e}], ...}."""
    words = []
    for w in data.get("words") or []:
        try:
            text = str(w.get("word", "")).strip()
            if text:
                words.append({"w": text, "s": round(float(w["start"]), 3), "e": round(float(w["end"]), 3)})
        except (KeyError, TypeError, ValueError):
            continue

    if not words:
        # провайдер не вернул слова — раскладываем текст сегментов по времени пропорционально длине слов
        for seg in data.get("segments") or []:
            try:
                s = float(seg["start"])
                e = float(seg["end"])
                toks = str(seg.get("text", "")).split()
            except (KeyError, TypeError, ValueError):
                continue
            if not toks or e <= s:
                continue
            total_chars = sum(len(t) for t in toks) or 1
            t0 = s
            for tok in toks:
                dt = (e - s) * len(tok) / total_chars
                words.append({"w": tok, "s": round(t0, 3), "e": round(t0 + dt, 3)})
                t0 += dt

    if duration is None:
        duration = float(data.get("duration") or 0) or (words[-1]["e"] if words else 0.0)

    return {
        "language": data.get("language"),
        "duration": duration,
        "text": data.get("text", ""),
        "words": words,
        "model": STT_MODEL,
    }


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("ru"),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Расшифровка аудио (mp3) со словами и таймкодами. Деньги списываются
    после успешного результата, по секундам аудио."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY не задан на сервере")

    data = await audio.read()
    if len(data) < 2000:
        raise HTTPException(status_code=400, detail="Файл слишком короткий или пустой")
    if len(data) > STT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Файл слишком большой (максимум 24 МБ). Расшифруйте часть записи.")

    if user_id not in UNLIMITED_USER_IDS:
        user = get_or_create_user(db, user_id)
        if (user.balance_usd or 0.0) < 0.001:
            raise HTTPException(status_code=402, detail="Недостаточно средств на балансе")

    lang = None if (language or "").lower() in ("", "auto") else language.lower()

    path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_stt.mp3")
    with open(path, "wb") as f:
        f.write(data)
    try:
        duration = _audio_duration(path)
        if duration is not None and duration > STT_MAX_SECONDS:
            raise HTTPException(status_code=413, detail="Запись длиннее часа. Расшифруйте её по частям.")
        async with _sem():
            result = await asyncio.to_thread(_call_openrouter_stt, path, lang)
    finally:
        if os.path.exists(path):
            os.remove(path)

    out = _normalize(result, duration)
    if not out["words"]:
        raise HTTPException(status_code=422, detail="Речь не найдена в этой записи.")

    if user_id not in UNLIMITED_USER_IDS:
        cost = round(max(out["duration"], STT_MIN_BILLABLE_SECONDS) * STT_PRICE_PER_SECOND_USD, 4)
        charge_feature(
            db, user_id, max(cost, 0.0001), feature="stt",
            units=round(out["duration"], 1), note=f"{len(out['words'])} слов",
        )
    return out
