"""
Avatar Studio — модуль биллинга.

Отвечает за:
- пополнение баланса пользователя через Crypto Pay (@CryptoBot) — оплата
  QR-кодом в любой валюте (в т.ч. рубли), расчёт в USDT/TON на Telegram-кошелёк,
  без необходимости открывать банковский счёт;
- списание с баланса за генерацию (посекундно, с коэффициентом наценки 50%
  сверх фактической стоимости GPU-времени на RunPod).

Требует переменную окружения CRYPTO_PAY_TOKEN — получить через @CryptoBot
командой /pay -> Create App.
"""
import os
import time
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

CRYPTO_PAY_TOKEN = os.getenv("CRYPTO_PAY_TOKEN")
CRYPTO_PAY_BASE_URL = "https://pay.crypt.bot/api"

# Наценка поверх фактической стоимости GPU-времени RunPod.
MARKUP_COEFFICIENT = 1.5  # +50%

# Ориентировочная стоимость GPU-времени в долларах за секунду.
# ВАЖНО: подставь сюда реальную ставку из RunPod (Billing -> Usage), это
# сейчас примерное значение под GPU 24GB, которое вы используете
# (RunPod указывал $0.69/hr в консоли -> $0.69 / 3600 сек).
RUNPOD_COST_PER_SECOND_USD = 0.69 / 3600

router = APIRouter(prefix="/api/billing", tags=["billing"])

CRYPTO_PAY_HEADERS = {
    "Crypto-Pay-API-Token": CRYPTO_PAY_TOKEN,
}


class CreateInvoiceRequest(BaseModel):
    user_id: str
    amount: float
    currency: str  # "RUB" или "USD" — фиатный эквивалент, сконвертируется в крипту автоматически


class InvoiceStatusRequest(BaseModel):
    invoice_id: int


def create_crypto_pay_invoice(amount: float, currency: str, description: str) -> dict:
    """Создаёт счёт в Crypto Pay. currency — фиатный код (RUB, USD и т.д.),
    Crypto Pay сам покажет плательщику сумму в USDT/TON по актуальному курсу.
    Возвращает словарь с полями invoice_id, pay_url (по нему строится QR)."""
    if not CRYPTO_PAY_TOKEN:
        raise HTTPException(status_code=500, detail="CRYPTO_PAY_TOKEN не настроен на сервере")

    payload = {
        "currency_type": "fiat",
        "fiat": currency,
        "amount": str(amount),
        "description": description,
        "paid_btn_name": "callback",
        "paid_btn_url": "https://t.me/Bestconsultingbot",
    }
    resp = requests.post(
        f"{CRYPTO_PAY_BASE_URL}/createInvoice",
        headers=CRYPTO_PAY_HEADERS,
        json=payload,
        timeout=15,
    )
    data = resp.json()
    if not data.get("ok"):
        raise HTTPException(status_code=502, detail=f"Crypto Pay вернул ошибку: {data}")
    return data["result"]


def check_crypto_pay_invoice(invoice_id: int) -> dict:
    """Проверяет статус счёта — 'active' (ещё не оплачен) или 'paid'."""
    resp = requests.get(
        f"{CRYPTO_PAY_BASE_URL}/getInvoices",
        headers=CRYPTO_PAY_HEADERS,
        params={"invoice_ids": str(invoice_id)},
        timeout=15,
    )
    data = resp.json()
    if not data.get("ok") or not data["result"]["items"]:
        raise HTTPException(status_code=404, detail="Счёт не найден")
    return data["result"]["items"][0]


def calculate_generation_cost(duration_seconds: float) -> float:
    """Считает стоимость генерации в USD: фактическая GPU-стоимость за
    время генерации, умноженная на коэффициент наценки (+50%)."""
    base_cost = duration_seconds * RUNPOD_COST_PER_SECOND_USD
    return round(base_cost * MARKUP_COEFFICIENT, 4)


@router.post("/create-deposit")
async def create_deposit(req: CreateInvoiceRequest):
    """Клиент выбирает сумму и валюту (RUB/USD) в приложении -> здесь
    создаётся счёт в Crypto Pay -> фронтенд строит QR-код из pay_url
    (любой библиотекой генерации QR на клиенте, например qrcode.js)."""
    invoice = create_crypto_pay_invoice(
        amount=req.amount,
        currency=req.currency,
        description=f"Пополнение баланса Avatar Studio (user {req.user_id})",
    )
    return {
        "invoice_id": invoice["invoice_id"],
        "pay_url": invoice["pay_url"],
        "amount": req.amount,
        "currency": req.currency,
    }


@router.post("/check-deposit")
async def check_deposit(req: InvoiceStatusRequest):
    """Опрашивается фронтендом после показа QR — как только status == 'paid',
    нужно зачислить сумму на баланс пользователя в базе данных (см. TODO
    ниже — подключить реальную БД, сейчас это заглушка без сохранения)."""
    invoice = check_crypto_pay_invoice(req.invoice_id)
    is_paid = invoice["status"] == "paid"

    # TODO: как только появится БД (Фаза 2 из плана) — здесь нужно:
    # 1) проверить, что этот invoice_id ещё не был зачислен ранее (защита
    #    от повторного начисления при многократном опросе)
    # 2) прибавить invoice["amount"] к балансу пользователя, сохранив
    #    invoice_id как обработанный

    return {
        "invoice_id": req.invoice_id,
        "status": invoice["status"],
        "paid": is_paid,
    }


@router.get("/estimate")
async def estimate_cost(duration_seconds: float):
    """Утилитарный эндпоинт — можно вызвать с фронтенда, чтобы показать
    клиенту ориентировочную стоимость до генерации (например, по длине
    введённого текста / загруженного аудио)."""
    cost = calculate_generation_cost(duration_seconds)
    return {
        "duration_seconds": duration_seconds,
        "estimated_cost_usd": cost,
    }
