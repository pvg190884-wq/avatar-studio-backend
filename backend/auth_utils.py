"""
Avatar Studio — проверка авторизации через Supabase Auth.

Фронтенд после входа/регистрации через Supabase получает access_token —
клиент передаёт его в заголовке Authorization: Bearer <token> при каждом
запросе к нашему API. Этот модуль проверяет токен напрямую у Supabase и
возвращает реальный, проверенный id пользователя — вместо того чтобы
слепо доверять произвольной строке user_id, которую клиент мог бы прислать
сам (как было раньше в billing.py, до подключения реальной авторизации).

Переменные окружения:
  SUPABASE_URL             — Project URL из настроек Supabase
  SUPABASE_PUBLISHABLE_KEY — Publishable key (безопасен для фронтенда)
"""
import os
import requests
from fastapi import Header, HTTPException

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")


async def get_current_user_id(authorization: str = Header(...)) -> str:
    """FastAPI-зависимость: подключается к защищённым эндпоинтам через
    Depends(get_current_user_id). Проверяет access_token у Supabase и
    возвращает id пользователя (UUID из Supabase) — использовать именно
    его как user_id в таблицах users/deposits, а не то, что прислал бы
    клиент сам по себе."""
    if not SUPABASE_URL or not SUPABASE_PUBLISHABLE_KEY:
        raise HTTPException(status_code=500, detail="Supabase не настроен на сервере")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Ожидается заголовок Authorization: Bearer <token>")

    token = authorization.removeprefix("Bearer ").strip()

    resp = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "apikey": SUPABASE_PUBLISHABLE_KEY,
            "Authorization": f"Bearer {token}",
        },
        timeout=10,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Недействительный или истёкший токен авторизации")

    user_data = resp.json()
    return user_data["id"]  # UUID пользователя в Supabase — стабильный, уникальный идентификатор
