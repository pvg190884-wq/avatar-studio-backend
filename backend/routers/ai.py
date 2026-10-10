"""
Avatar Studio — ИИ-шлюз для Video Studio: расшифровка речи и план монтажа через OpenRouter.

Ключ OpenRouter хранится только здесь (переменная окружения), в браузер он
не уходит. Оплата — с общего баланса (billing.charge_user), после успешного
результата. Вход проверяется через Supabase, как у остальных эндпоинтов.

Переменные окружения:
  OPENROUTER_API_KEY        — ключ OpenRouter (уже задан на Railway)
  STT_MODEL                 — модель распознавания (по умолчанию openai/whisper-large-v3-turbo)
  STT_PRICE_PER_SECOND_USD  — цена для клиента за секунду аудио (по умолчанию 0.00005, ≈ $0.18 за час)
  EDIT_MODEL                — модель-монтажёр (по умолчанию openai/gpt-4o-mini)
  EDIT_MODEL_FALLBACK       — модель на случай непонятного ответа (по умолчанию openai/gpt-4o)
  EDIT_MIN_PRICE_USD        — минимальная цена команды монтажа (по умолчанию 0.005)
  EDIT_COST_MULTIPLIER      — множитель к фактической стоимости запроса (по умолчанию 3)
  UNLIMITED_USER_IDS        — id пользователей без списаний (как в runpod_avatar.py)
"""
import os
import re
import json
import uuid
import asyncio
import requests
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from mutagen import File as MutagenFile
from dotenv import load_dotenv

from database import get_db
from auth_utils import get_current_user_id
from routers.billing import get_or_create_user, charge_user as charge_feature

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_STT_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
STT_MODEL = os.getenv("STT_MODEL", "openai/whisper-large-v3-turbo")
# Цена для клиента за 1 секунду аудио. Стартовое значение: пересчитайте по факту
# списаний на openrouter.ai/activity и поменяйте переменной на Railway.
STT_PRICE_PER_SECOND_USD = float(os.getenv("STT_PRICE_PER_SECOND_USD", "0.00005"))
STT_MIN_BILLABLE_SECONDS = 5.0
STT_MAX_BYTES = 24 * 1024 * 1024
STT_MAX_SECONDS = 60 * 60

EDIT_MODEL = os.getenv("EDIT_MODEL", "openai/gpt-4o-mini")
EDIT_MODEL_FALLBACK = os.getenv("EDIT_MODEL_FALLBACK", "openai/gpt-4o")
EDIT_MIN_PRICE_USD = float(os.getenv("EDIT_MIN_PRICE_USD", "0.005"))
EDIT_COST_MULTIPLIER = float(os.getenv("EDIT_COST_MULTIPLIER", "3"))
EDIT_MAX_SEGMENTS = 600
EDIT_MAX_CHARS = 60000
EDIT_MAX_OPS = 8

UNLIMITED_USER_IDS = {u.strip() for u in os.getenv("UNLIMITED_USER_IDS", "").split(",") if u.strip()}

TEMP_DIR = "data/runpod_tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

router = APIRouter(prefix="/api/ai", tags=["ai"])

_stt_sem: Optional[asyncio.Semaphore] = None


def _sem() -> asyncio.Semaphore:
    """Не больше 4 тяжёлых запросов одновременно, чтобы не забить сервер."""
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


def _check_balance(db: Session, user_id: str, minimum: float):
    if user_id in UNLIMITED_USER_IDS:
        return
    user = get_or_create_user(db, user_id)
    if (user.balance_usd or 0.0) < minimum:
        raise HTTPException(status_code=402, detail="Недостаточно средств на балансе")


# ======================================================================
# Расшифровка речи
# ======================================================================
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

    _check_balance(db, user_id, 0.001)

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


# ======================================================================
# ИИ-монтажёр: команда -> план из разрешённых операций
# ======================================================================
FONTS = {"Manrope", "Montserrat", "Oswald", "Playfair Display", "Lora", "Pacifico", "Caveat", "Russo One"}
COVER_TPLS = {"gradient", "minimal", "neon", "cinema", "split", "glitch"}
COLOR_PRESETS = {"none", "cinema", "warm", "cold", "vintage", "vivid", "faded", "bw", "noir", "sepia"}
SUB_BG = {"shadow", "outline", "box", "none"}
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

EDIT_SYSTEM_PROMPT = """You are the editing assistant inside a browser video editor. The user gives a command in natural language (usually Russian). You reply with ONE JSON object and nothing else:
{"message": "<short reply to the user, in the user's language>", "ops": [ ... ]}

You cannot touch the video directly. You may only use these operations in "ops" (in the order they should be applied; use as few as needed):
- {"op":"cut","segments":[ids]}  remove these transcript segments from the video.
- {"op":"keep","segments":[ids]}  keep ONLY these segments and remove everything else. Use it for "make an N-second video", "keep only the part about X", highlights. Choose complete thoughts in chronological order, and make the total duration of the kept segments close to the requested length (durations are given).
- {"op":"fillers","sounds":true,"words":false}  remove hesitation sounds (and, if words=true, filler words like "типа", "короче").
- {"op":"subtitles","size":3-12,"per":1-8,"color":"#rrggbb","hl":"#rrggbb" or "","bg":"shadow|outline|box|none","y":0.1-0.95,"upper":true|false,"font":"Manrope|Montserrat|Oswald|Playfair Display|Lora|Pacifico|Caveat|Russo One"}  add or restyle subtitles; every field is optional.
- {"op":"cover","tpl":"gradient|minimal|neon|cinema|split|glitch","title":"...","sub":"...","where":"start|end"}  add a 4-second animated title card.
- {"op":"color","preset":"none|cinema|warm|cold|vintage|vivid|faded|bw|noir|sepia","strength":0-1}  colour-grade all clips.

Rules:
- Use only segment ids that appear in the transcript. Never invent ids or times.
- If the transcript is empty, do not use cut, keep or fillers.
- If the command cannot be done with these operations, return "ops": [] and explain in "message" what is possible instead.
- Do not remove more than needed. When unsure, prefer a conservative edit and say so in "message".
- "message" is short (1-3 sentences) and describes what the ops will do. Never claim something that is not in "ops".
- Output valid JSON only, no markdown."""


class EditSeg(BaseModel):
    id: int = 0
    s: float = 0.0
    e: float = 0.0
    text: str = ""


class EditRequest(BaseModel):
    command: str
    segments: list[EditSeg] = []
    duration: float = 0.0
    has_subs: bool = False


def _chat(model: str, system: str, user: str) -> tuple[str, dict]:
    """Один запрос к OpenRouter Chat Completions. Только через asyncio.to_thread."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "BestConsulting Video Studio",
    }
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
        "usage": {"include": True},
    }
    try:
        resp = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=body, timeout=90)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Не удалось связаться с ИИ-сервисом: {e}")
    if resp.status_code == 402:
        print(f"OpenRouter chat: 402 Payment Required — пополните кредиты OpenRouter. {_error_text(resp)}")
        raise HTTPException(status_code=503, detail="ИИ-монтажёр временно недоступен. Попробуйте позже.")
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"ИИ-сервис ответил ошибкой: {_error_text(resp)}")
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        content = ""
    return content, (data.get("usage") or {})


def _parse_json(text: str):
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(t[a:b + 1])
        except Exception:
            return None
    return None


def _num(v, lo: float, hi: float):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return min(hi, max(lo, x))


def _ids(v, n: int) -> list[int]:
    out: list[int] = []
    if isinstance(v, list):
        for x in v:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= i < n and i not in out:
                out.append(i)
    return sorted(out)


def _clean_plan(parsed, n: int):
    """Оставляет только разрешённые операции с проверенными параметрами.
    Возвращает None, если ответ не похож на план."""
    if not isinstance(parsed, dict) or ("ops" not in parsed and "message" not in parsed):
        return None
    ops = []
    raw_ops = parsed.get("ops")
    for o in (raw_ops if isinstance(raw_ops, list) else [])[:EDIT_MAX_OPS]:
        if not isinstance(o, dict):
            continue
        kind = o.get("op")
        if kind in ("cut", "keep"):
            ids = _ids(o.get("segments"), n)
            if ids:
                ops.append({"op": kind, "segments": ids})
        elif kind == "fillers":
            ops.append({"op": "fillers", "sounds": o.get("sounds") is not False, "words": bool(o.get("words"))})
        elif kind == "subtitles":
            p = {"op": "subtitles"}
            size = _num(o.get("size"), 3, 12)
            if size is not None:
                p["size"] = size
            per = _num(o.get("per"), 1, 8)
            if per is not None:
                p["per"] = int(round(per))
            if isinstance(o.get("color"), str) and HEX.match(o["color"]):
                p["color"] = o["color"]
            if o.get("hl") == "" or (isinstance(o.get("hl"), str) and HEX.match(o["hl"])):
                p["hl"] = o["hl"]
            if o.get("bg") in SUB_BG:
                p["bg"] = o["bg"]
            y = _num(o.get("y"), 0.1, 0.95)
            if y is not None:
                p["y"] = y
            if isinstance(o.get("upper"), bool):
                p["upper"] = o["upper"]
            if o.get("font") in FONTS:
                p["font"] = o["font"]
            ops.append(p)
        elif kind == "cover":
            if o.get("tpl") in COVER_TPLS:
                ops.append({
                    "op": "cover", "tpl": o["tpl"],
                    "title": str(o.get("title") or "")[:80], "sub": str(o.get("sub") or "")[:80],
                    "where": "end" if o.get("where") == "end" else "start",
                })
        elif kind == "color":
            if o.get("preset") in COLOR_PRESETS:
                st = _num(o.get("strength"), 0, 1)
                ops.append({"op": "color", "preset": o["preset"], "strength": 1.0 if st is None else st})
    return {"message": str(parsed.get("message") or "")[:600], "ops": ops}


@router.post("/edit")
async def edit_plan(
    req: EditRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """ИИ-монтажёр: по команде и расшифровке возвращает план из разрешённых операций.
    Видео модель не видит и ничего не меняет сама: план применяет и проверяет клиент.
    Деньги списываются после успешного плана."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY не задан на сервере")

    command = (req.command or "").strip()
    if not command:
        raise HTTPException(status_code=400, detail="Напишите команду для монтажа")
    if len(command) > 500:
        raise HTTPException(status_code=400, detail="Команда слишком длинная (максимум 500 символов)")

    segs = req.segments
    if len(segs) > EDIT_MAX_SEGMENTS or sum(len(s.text) for s in segs) > EDIT_MAX_CHARS:
        raise HTTPException(status_code=413, detail="Текст расшифровки слишком длинный для одного запроса. Работайте с частью проекта.")

    _check_balance(db, user_id, 0.01)

    # номер фрагмента для модели = позиция в списке (клиент использует те же номера)
    lines = [
        f"{i} [{s.s:.1f}-{s.e:.1f}s, {max(0.0, s.e - s.s):.1f}s] {s.text.strip()[:400]}"
        for i, s in enumerate(segs)
    ]
    user_prompt = (
        f"Command: {command}\n"
        f"Current video duration: {req.duration:.1f} s. Subtitles already on the timeline: {'yes' if req.has_subs else 'no'}.\n"
        + ("Transcript segments (id [start-end, duration] text):\n" + "\n".join(lines) if lines else "Transcript: (empty)")
    )

    used = EDIT_MODEL
    cost = 0.0
    async with _sem():
        content, usage = await asyncio.to_thread(_chat, EDIT_MODEL, EDIT_SYSTEM_PROMPT, user_prompt)
        cost += float(usage.get("cost") or 0)
        plan = _clean_plan(_parse_json(content), len(segs))
        if plan is None and EDIT_MODEL_FALLBACK and EDIT_MODEL_FALLBACK != EDIT_MODEL:
            content, usage = await asyncio.to_thread(_chat, EDIT_MODEL_FALLBACK, EDIT_SYSTEM_PROMPT, user_prompt)
            cost += float(usage.get("cost") or 0)
            plan = _clean_plan(_parse_json(content), len(segs))
            used = EDIT_MODEL_FALLBACK

    if plan is None:
        raise HTTPException(status_code=502, detail="ИИ не вернул понятный план. Переформулируйте команду. Деньги не списаны.")

    price = round(max(EDIT_MIN_PRICE_USD, cost * EDIT_COST_MULTIPLIER) if cost > 0 else EDIT_MIN_PRICE_USD * 2, 4)
    if user_id not in UNLIMITED_USER_IDS:
        charge_feature(
            db, user_id, price, feature="ai-edit",
            units=1, note=f"{used}, {len(segs)} фрагм.",
        )
    return {"message": plan["message"], "ops": plan["ops"], "model": used, "price_usd": price}
