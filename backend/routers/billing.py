"""
Avatar Studio — модуль биллинга.

Отвечает за:
- пополнение баланса через Crypto Pay (@CryptoBot) — QR-оплата в RUB/USD/
  BTC/TON/USDT. Зачисление баланса срабатывает ДВУМЯ путями: (1) вебхук
  от Crypto Pay напрямую на сервер в момент оплаты — основной, надёжный
  путь, не зависящий от того, открыта ли у клиента вкладка браузера;
  (2) опрос /check-deposit из браузера — подстраховка/быстрый визуальный
  отклик, если вебхук ещё не настроен или чуть задержался;
- пополнение через СБП (личный перевод по номеру телефона) — при
  создании заявки админу мгновенно приходит сообщение в Telegram со
  ссылкой на подтверждение в один клик, без Swagger/curl;
- списание с баланса за генерацию (посекундно, +50% наценка к стоимости
  GPU-времени RunPod).

Переменные окружения:
  CRYPTO_PAY_TOKEN               — получить через @CryptoBot командой /pay -> Create App
  CRYPTO_PAY_WEBHOOK_SECRET_PATH — произвольная секретная строка для URL вебхука
                                    (см. настройку ниже, в комментарии к /crypto-webhook)
  ADMIN_SECRET                   — произвольная строка-пароль для подтверждения СБП-заявок
  SBP_PHONE                      — номер телефона для личных переводов по СБП
  TELEGRAM_BOT_TOKEN             — токен бота (у @BotFather), для уведомлений админу
  ADMIN_TELEGRAM_CHAT_ID         — твой личный Telegram chat_id (куда слать уведомления)
  BACKEND_PUBLIC_URL             — публичный адрес этого бэкенда (для ссылок в уведомлениях)
  DATABASE_URL                   — создаётся автоматически Railway при добавлении Postgres
"""
import os
import json
import hmac
import hashlib
import asyncio
import requests
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, User, Deposit, init_db
from auth_utils import get_current_user_id

CRYPTO_PAY_TOKEN = os.getenv("CRYPTO_PAY_TOKEN")
CRYPTO_PAY_BASE_URL = "https://pay.crypt.bot/api"
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
SBP_PHONE = os.getenv("SBP_PHONE", "не настроен")

# Секретный сегмент пути для вебхука Crypto Pay — доп. защита сверх
# проверки подписи (так рекомендует сама документация Crypto Pay:
# "we recommend using a secret path in the URL"). Придумай длинную
# случайную строку и укажи её и здесь, и при включении вебхука в
# настройках приложения @CryptoBot.
CRYPTO_PAY_WEBHOOK_SECRET_PATH = os.getenv("CRYPTO_PAY_WEBHOOK_SECRET_PATH", "")

# Для мгновенных уведомлений админу о новых заявках СБП в Telegram.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_TELEGRAM_CHAT_ID = os.getenv("ADMIN_TELEGRAM_CHAT_ID")
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "https://avatar-studio-backend-production.up.railway.app")

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
RUB_TO_USD_RATE = 0.010204  # ориентировочно, ~98 руб. за доллар

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


def credit_deposit_if_pending(db: Session, invoice: dict) -> bool:
    """Общая логика зачисления по оплаченному инвойсу Crypto Pay —
    используется и из вебхука (см. /crypto-webhook), и из ручного
    опроса /check-deposit из браузера. Идемпотентна: если депозит уже
    не в статусе pending (например, оба пути сработали почти
    одновременно), просто ничего не делает — защита от двойного
    зачисления. .with_for_update() блокирует строку на время
    транзакции, чтобы исключить гонку между вебхуком и опросом."""
    invoice_id = invoice.get("invoice_id")
    if invoice_id is None or invoice.get("status") != "paid":
        return False

    deposit = db.query(Deposit).filter(
        Deposit.external_id == str(invoice_id),
        Deposit.status == "pending",
    ).with_for_update().first()
    if not deposit:
        return False

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
    return True


def verify_crypto_pay_signature(raw_body: bytes, signature_header: str) -> bool:
    """Проверка подлинности вебхука — см. документацию Crypto Pay:
    подпись — это hex HMAC-SHA256 от тела запроса (как есть, без
    парсинга), ключ — SHA256-хэш токена приложения."""
    if not CRYPTO_PAY_TOKEN or not signature_header:
        return False
    secret_key = hashlib.sha256(CRYPTO_PAY_TOKEN.encode()).digest()
    computed = hmac.new(secret_key, raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature_header)


def send_telegram_message(chat_id: str, text: str):
    """ВАЖНО: это синхронная блокирующая функция (requests.post). Она
    ДОЛЖНА вызываться только через asyncio.to_thread() из async-кода —
    иначе, пока Telegram API отвечает (вплоть до timeout=10 секунд),
    единственный поток event loop FastAPI полностью замирает, и сервер
    перестаёт отвечать вообще на ВСЕ запросы от ВСЕХ пользователей, а
    не только на этот один (тот же класс бага, что раньше уже был
    найден и исправлен в runpod_avatar.py для вызовов RunPod — здесь
    он был пропущен). См. историю чата: именно это вызвало 10-минутное
    зависание заявки СБП и одновременный сбой у другого пользователя,
    никак не связанного с СБП."""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        print("TELEGRAM_BOT_TOKEN/ADMIN_TELEGRAM_CHAT_ID не настроены — уведомление не отправлено")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=10,
        )
    except requests.exceptions.RequestException as e:
        print(f"Не удалось отправить Telegram-уведомление: {e}")


def make_sbp_confirm_token(deposit_id: int) -> str:
    """Короткий токен, привязанный к конкретной заявке (не универсальный
    пароль) — вычисляется из ADMIN_SECRET, так что ссылку можно смело
    отправлять в Telegram: она работает только для этой одной заявки,
    а сам ADMIN_SECRET в ней не фигурирует."""
    return hmac.new(ADMIN_SECRET.encode(), str(deposit_id).encode(), hashlib.sha256).hexdigest()[:16]


def notify_admin_new_sbp_request(deposit_id: int, amount_rub: float, user_id: str):
    token = make_sbp_confirm_token(deposit_id)
    confirm_url = f"{BACKEND_PUBLIC_URL}/api/billing/sbp/confirm-link?deposit_id={deposit_id}&token={token}"
    text = (
        f"\U0001F4B0 Новая заявка СБП\n"
        f"Сумма: {amount_rub} \u20bd\n"
        f"Клиент: {user_id}\n\n"
        f"Подтвердить зачисление (один клик):\n{confirm_url}"
    )
    send_telegram_message(ADMIN_TELEGRAM_CHAT_ID, text)


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
    """Опрашивается фронтендом после показа QR — подстраховка/быстрый
    визуальный отклик. Основное зачисление теперь происходит через
    вебхук (см. /crypto-webhook) и не зависит от того, открыта ли эта
    вкладка."""
    invoice = check_crypto_pay_invoice(req.invoice_id)
    is_paid = invoice["status"] == "paid"

    if is_paid:
        credit_deposit_if_pending(db, invoice)

    return {
        "invoice_id": req.invoice_id,
        "status": invoice["status"],
        "paid": is_paid,
    }


@router.post("/crypto-webhook/{secret_path}")
async def crypto_pay_webhook(secret_path: str, request: Request, db: Session = Depends(get_db)):
    """Вебхук от Crypto Pay (@CryptoBot) — приходит НАПРЯМУЮ на сервер
    в момент оплаты, независимо от того, открыта ли у клиента вкладка
    браузера. Раньше зачисление зависело только от опроса
    /check-deposit из браузера — если клиент закрывал вкладку до
    подтверждения оплаты, баланс не зачислялся вообще, хотя деньги уже
    приходили на @CryptoBot. Вебхук устраняет эту зависимость полностью.

    НАСТРОЙКА (один раз, вручную, в Telegram):
      1. Открой @CryptoBot -> Crypto Pay -> My Apps -> твоё приложение
      2. Найди раздел Webhooks -> Enable
      3. URL вебхука:
         https://<домен бэкенда>/api/billing/crypto-webhook/<CRYPTO_PAY_WEBHOOK_SECRET_PATH>
         (значение секретного пути — то же, что в переменной окружения
         Railway CRYPTO_PAY_WEBHOOK_SECRET_PATH, придумай длинную
         случайную строку)
    """
    if not CRYPTO_PAY_WEBHOOK_SECRET_PATH or secret_path != CRYPTO_PAY_WEBHOOK_SECRET_PATH:
        raise HTTPException(status_code=404)

    raw_body = await request.body()
    signature = request.headers.get("crypto-pay-api-signature", "")
    if not verify_crypto_pay_signature(raw_body, signature):
        raise HTTPException(status_code=401, detail="Неверная подпись вебхука")

    try:
        data = json.loads(raw_body)
    except Exception:
        raise HTTPException(status_code=400, detail="Некорректное тело запроса")

    # Crypto Pay может присылать разные типы обновлений — интересует
    # только оплата счёта.
    if data.get("update_type") == "invoice_paid":
        invoice = data.get("payload", {})
        credit_deposit_if_pending(db, invoice)

    return {"ok": True}


@router.post("/sbp/request")
async def sbp_request(
    req: SbpRequestBody,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Клиент выбрал способ оплаты СБП -> создаётся заявка 'pending' ->
    фронтенд показывает номер телефона (SBP_PHONE). Админу мгновенно
    приходит уведомление в Telegram со ссылкой на подтверждение в один
    клик (см. /sbp/confirm-link) — без Swagger/curl.

    Уведомление отправляется через asyncio.to_thread — см. подробный
    комментарий у send_telegram_message() про то, почему синхронный
    requests.post() здесь НЕЛЬЗЯ вызывать напрямую."""
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

    await asyncio.to_thread(notify_admin_new_sbp_request, deposit.id, req.amount_rub, user_id)

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


@router.get("/sbp/confirm-link", response_class=HTMLResponse)
async def sbp_confirm_link(deposit_id: int, token: str, db: Session = Depends(get_db)):
    """Ссылка из Telegram-уведомления — одна ссылка подтверждает ровно
    одну заявку. Открывается прямо в браузере (в том числе с телефона,
    прямо из Telegram), никакого Swagger/curl/ручного ввода
    ADMIN_SECRET не требуется."""
    expected_token = make_sbp_confirm_token(deposit_id)
    if not hmac.compare_digest(token, expected_token):
        return HTMLResponse("<h3>Ссылка неверна или устарела</h3>", status_code=403)

    deposit = db.query(Deposit).filter(Deposit.id == deposit_id).with_for_update().first()
    if not deposit:
        return HTMLResponse("<h3>Заявка не найдена</h3>", status_code=404)
    if deposit.status != "pending":
        return HTMLResponse(f"<h3>Заявка уже обработана (статус: {deposit.status})</h3>")

    amount_usd = deposit.amount * RUB_TO_USD_RATE
    user = get_or_create_user(db, deposit.user_id)
    user.balance_usd += amount_usd
    deposit.status = "confirmed"
    deposit.amount_usd = amount_usd
    deposit.confirmed_at = datetime.utcnow()
    db.commit()

    return HTMLResponse(
        f"<h2>\u2705 Зачислено ${amount_usd:.2f} пользователю {deposit.user_id}</h2>"
        f"<p>Сумма: {deposit.amount} ₽ &rarr; ${amount_usd:.2f}</p>"
    )


@router.post("/sbp/confirm")
async def sbp_confirm(req: SbpConfirmBody, db: Session = Depends(get_db)):
    """Запасной путь подтверждения СБП-заявки через admin_secret (на
    случай, если Telegram-уведомление не пришло/потерялось) — тот же
    механизм, что и раньше."""
    if not ADMIN_SECRET or req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Неверный admin_secret")

    deposit = db.query(Deposit).filter(Deposit.id == req.deposit_id).with_for_update().first()
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
