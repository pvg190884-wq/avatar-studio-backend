"""
Avatar Studio — модуль биллинга.

Отвечает за:
- пополнение баланса через Crypto Pay (@CryptoBot) — QR-оплата в RUB/USD/
  BTC/TON/USDT, без банковского счёта;
- пополнение через СБП (личный перевод по номеру телефона) — полу-ручной
  режим: клиент переводит и указывает номер операции, администратор
  подтверждает вручную через защищённый эндпоинт;
- списание с баланса за генерацию (посекундно, +50% наценка к стоимости
  GPU-времени RunPod).

Переменные окружения:
  CRYPTO_PAY_TOKEN — получить через @CryptoBot командой /pay -> Create App
  ADMIN_SECRET     — произвольная строка-пароль для подтверждения СБП-заявок
  SBP_PHONE        — номер телефона для личных переводов по СБП
  DATABASE_URL     — создаётся автоматически Railway при добавлении Postgres
"""
import os
import requests
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, User, Deposit, init_db
from auth_utils import get_current_user_id

CRYPTO_PAY_TOKEN = os.getenv("CRYPTO_PAY_TOKEN")
CRYPTO_PAY_BASE_URL = "https://pay.crypt.bot/api"
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
SBP_PHONE = os.getenv("SBP_PHONE", "не настроен")

# Наценка поверх фактической стоимости GPU-времени RunPod.
MARKUP_COEFFICIENT = 1.5  # +50%

# Реальная стоимость GPU-времени в долларах за секунду — по факту
# замера, а не по прикидке "$/час из карточки GPU на сайте RunPod".
# Как пересчитать при изменении типа GPU / тарифа RunPod:
#   1. Взять из RunPod Dashboard -> Billing -> Usage фактическую сумму,
#      списанную за некоторое количество завершённых генераций.
#   2. Взять суммарную длительность аудио (не видео!) по тем же
#      генерациям — именно секунды аудио, а не время работы GPU,
#      являются единицей тарификации на сайте (см. calculate_generation_cost).
#   3. RUNPOD_COST_PER_SECOND_USD = (сумма из шага 1) / (сумма из шага 2).
# Текущее значение 0.009 — по замеру от 2026-09 (см. чат): реальная
# стоимость для владельца составила $0.009 за 1 секунду аудио.
# Цена для клиента = RUNPOD_COST_PER_SECOND_USD * MARKUP_COEFFICIENT
#                   = 0.009 * 1.5 = $0.0135 за секунду.
RUNPOD_COST_PER_SECOND_USD = 0.009

# Курс для конвертации рублёвых СБП-пополнений в USD-баланс.
# ВАЖНО: это фиксированное приближение, не биржевой курс в реальном
# времени — обновляй вручную по мере необходимости, либо замени на
# запрос к какому-нибудь бесплатному API курсов валют, когда дойдут руки.
RUB_TO_USD_RATE = 0.011  # ориентировочно, ~90 руб. за доллар

router = APIRouter(prefix="/api/billing", tags=["billing"])

CRYPTO_PAY_HEADERS = {
    "Crypto-Pay-API-Token": CRYPTO_PAY_TOKEN,
}

# Способы оплаты в окне приложения. RUB/USD — фиатное отображение суммы
# через Crypto Pay (платёж всё равно в крипте, но список монет сужен через
# accepted_assets). BTC/TON/USDT — прямой выбор монеты. SBP — отдельный
# полу-ручной путь, не через Crypto Pay вообще.
FIAT_METHODS = {"RUB", "USD"}
CRYPTO_METHODS = {"BTC", "TON", "USDT"}
ALL_CRYPTO_PAY_METHODS = FIAT_METHODS | CRYPTO_METHODS
ACCEPTED_ASSETS_FOR_FIAT = "USDT,TON,BTC"


class CreateInvoiceRequest(BaseModel):
    amount: float
    method: str  # "RUB", "USD", "BTC", "TON", "USDT"


class InvoiceStatusRequest(BaseModel):
    invoice_id: int


class SbpRequestBody(BaseModel):
    amount_rub: float


class SbpConfirmBody(BaseModel):
    deposit_id: int
    admin_secret: str
    approve: bool  # True — подтвердить и зачислить, False — отклонить


def get_or_create_user(db: Session, user_id: str) -> User:
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(user_id=user_id, balance_usd=0.0)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def create_crypto_pay_invoice(amount: float, method: str, description: str) -> dict:
    if not CRYPTO_PAY_TOKEN:
        raise HTTPException(status_code=500, detail="CRYPTO_PAY_TOKEN не настроен на сервере")

    method = method.upper()
    if method not in ALL_CRYPTO_PAY_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный способ оплаты '{method}', доступны: {sorted(ALL_CRYPTO_PAY_METHODS)}",
        )

    if method in FIAT_METHODS:
        payload = {
            "currency_type": "fiat",
            "fiat": method,
            "amount": str(amount),
            "accepted_assets": ACCEPTED_ASSETS_FOR_FIAT,
            "description": description,
            "paid_btn_name": "callback",
            "paid_btn_url": "https://t.me/Bestconsultingbot",
        }
    else:
        payload = {
            "currency_type": "crypto",
            "asset": method,
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
    base_cost = duration_seconds * RUNPOD_COST_PER_SECOND_USD
    return round(base_cost * MARKUP_COEFFICIENT, 4)


@router.post("/create-deposit")
async def create_deposit(
    req: CreateInvoiceRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Крипто-путь (RUB/USD/BTC/TON/USDT через Crypto Pay)."""
    get_or_create_user(db, user_id)
    invoice = create_crypto_pay_invoice(
        amount=req.amount,
        method=req.method,
        description=f"Пополнение баланса Avatar Studio (user {user_id})",
    )

    deposit = Deposit(
        user_id=user_id,
        method=req.method.upper(),
        amount=req.amount,
        status="pending",
        external_id=str(invoice["invoice_id"]),
    )
    db.add(deposit)
    db.commit()

    return {
        "invoice_id": invoice["invoice_id"],
        "pay_url": invoice["pay_url"],
        "amount": req.amount,
        "method": req.method.upper(),
    }


@router.post("/check-deposit")
async def check_deposit(req: InvoiceStatusRequest, db: Session = Depends(get_db)):
    """Опрашивается фронтендом после показа QR. При первом обнаружении
    статуса 'paid' — зачисляет баланс и помечает депозит обработанным,
    защищено от повторного начисления при повторных опросах."""
    invoice = check_crypto_pay_invoice(req.invoice_id)
    is_paid = invoice["status"] == "paid"

    if is_paid:
        deposit = db.query(Deposit).filter(
            Deposit.external_id == str(req.invoice_id),
            Deposit.status == "pending",
        ).first()
        if deposit:
            amount_usd = float(invoice.get("amount", deposit.amount))
            if deposit.method == "RUB":
                amount_usd = deposit.amount * RUB_TO_USD_RATE
            elif deposit.method == "USD":
                amount_usd = deposit.amount
            # Для BTC/TON/USDT используем сумму, подтверждённую Crypto Pay,
            # как приближение к USD (грубо для BTC/TON, точно для USDT).

            user = get_or_create_user(db, deposit.user_id)
            user.balance_usd += amount_usd
            deposit.status = "confirmed"
            deposit.amount_usd = amount_usd
            deposit.confirmed_at = datetime.utcnow()
            db.commit()

    return {
        "invoice_id": req.invoice_id,
        "status": invoice["status"],
        "paid": is_paid,
    }


@router.post("/sbp/request")
async def sbp_request(
    req: SbpRequestBody,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Клиент выбрал способ оплаты СБП -> создаётся заявка 'pending' ->
    фронтенд показывает номер телефона (SBP_PHONE) и просит после
    перевода прислать номер операции. Зачисление — только после ручного
    подтверждения через /sbp/confirm."""
    get_or_create_user(db, user_id)

    deposit = Deposit(
        user_id=user_id,
        method="SBP",
        amount=req.amount_rub,
        status="pending",
    )
    db.add(deposit)
    db.commit()
    db.refresh(deposit)

    return {
        "deposit_id": deposit.id,
        "amount_rub": req.amount_rub,
        "sbp_phone": SBP_PHONE,
        "instructions": (
            f"Переведите {req.amount_rub} ₽ по СБП на номер {SBP_PHONE}. "
            f"После перевода сохраните номер операции — он понадобится "
            f"для подтверждения зачисления."
        ),
    }


@router.post("/sbp/confirm")
async def sbp_confirm(req: SbpConfirmBody, db: Session = Depends(get_db)):
    """Ручное подтверждение СБП-заявки администратором. Защищено простым
    секретным паролем (ADMIN_SECRET) — временное решение до появления
    полноценной админ-авторизации."""
    if not ADMIN_SECRET or req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Неверный admin_secret")

    deposit = db.query(Deposit).filter(Deposit.id == req.deposit_id).first()
    if not deposit:
        raise HTTPException(status_code=404, detail="Заявка не найдена")
    if deposit.status != "pending":
        raise HTTPException(status_code=400, detail=f"Заявка уже обработана (статус: {deposit.status})")

    if req.approve:
        amount_usd = deposit.amount * RUB_TO_USD_RATE
        user = get_or_create_user(db, deposit.user_id)
        user.balance_usd += amount_usd
        deposit.status = "confirmed"
        deposit.amount_usd = amount_usd
    else:
        deposit.status = "rejected"

    deposit.confirmed_at = datetime.utcnow()
    db.commit()

    return {"deposit_id": deposit.id, "status": deposit.status}


@router.get("/balance")
async def get_balance(db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Текущий баланс авторизованного пользователя в USD."""
    user = get_or_create_user(db, user_id)
    return {"user_id": user_id, "balance_usd": round(user.balance_usd, 4)}


@router.get("/estimate")
async def estimate_cost(duration_seconds: float):
    cost = calculate_generation_cost(duration_seconds)
    return {
        "duration_seconds": duration_seconds,
        "estimated_cost_usd": cost,
    }
