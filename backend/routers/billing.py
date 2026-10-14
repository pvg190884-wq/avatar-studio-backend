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
  создании заявки админу мгновенно приходит сообщение в Telegram и/или
  MAX со ссылкой на подтверждение в один клик, без Swagger/curl;
- списание с баланса за генерацию (посекундно, +50% наценка к стоимости
  GPU-времени RunPod);
- единый баланс для Video Studio: журнал списаний, история операций и
  подписка Video Studio (199 ₽ в месяц, оплата с общего баланса);
- информационные уведомления админу: подписка оформлена/продлена,
  пополнение через Crypto Pay зачислено.

Переменные окружения:
  CRYPTO_PAY_TOKEN               — получить через @CryptoBot командой /pay -> Create App
  CRYPTO_PAY_WEBHOOK_SECRET_PATH — произвольная секретная строка для URL вебхука
                                    (см. настройку ниже, в комментарии к /crypto-webhook)
  ADMIN_SECRET                   — произвольная строка-пароль для подтверждения СБП-заявок
  SBP_PHONE                      — номер телефона для личных переводов по СБП
  TELEGRAM_BOT_TOKEN             — токен бота (у @BotFather), для уведомлений админу
  ADMIN_TELEGRAM_CHAT_ID         — твой личный Telegram chat_id (куда слать уведомления)
  MAX_API_TOKEN                  — токен бота в MAX (платформа MAX для партнёров -> Интеграция)
  MAX_API_URL                    — адрес API MAX (как в боте bestconsulting-AI, без /messages на конце)
  MAX_ADMIN_USER_ID              — твой MAX user_id (по умолчанию 111620312)
  MAX_ADMIN_CHAT_ID              — необязательно: chat_id диалога с ботом; если задан, используется вместо user_id
  NOTIFY_CHANNELS                — каналы уведомлений через запятую: telegram,max (по умолчанию оба)
  NOTIFY_INFO                    — 0 отключает информационные сообщения (подписки, Crypto Pay); по умолчанию включены
  BACKEND_PUBLIC_URL             — публичный адрес этого бэкенда (для ссылок в уведомлениях)
  DATABASE_URL                   — создаётся автоматически Railway при добавлении Postgres
"""
import os
import json
import hmac
import math
import hashlib
import asyncio
import requests
import urllib3
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, User, Deposit, Charge, Subscription, init_db
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
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "https://api.avatar-studio.bestconsulting-usa.ru").rstrip("/")

# Уведомления в MAX (тот же бот, что в bestconsulting-AI). Получатель —
# личный диалог владельца: по user_id (по умолчанию founder ID из Core)
# или, если задан MAX_ADMIN_CHAT_ID, по chat_id.
MAX_API_TOKEN = os.getenv("MAX_API_TOKEN")
_max_url = (os.getenv("MAX_API_URL") or "https://platform-api.max.ru").strip().rstrip("/")
if _max_url.endswith("/messages"):
    _max_url = _max_url[: -len("/messages")]
MAX_API_URL = _max_url
MAX_ADMIN_USER_ID = os.getenv("MAX_ADMIN_USER_ID", "111620312")
MAX_ADMIN_CHAT_ID = os.getenv("MAX_ADMIN_CHAT_ID")

# В какие каналы слать уведомления админу: telegram, max или оба.
NOTIFY_CHANNELS = {c.strip().lower() for c in os.getenv("NOTIFY_CHANNELS", "telegram,max").split(",") if c.strip()}

# Информационные сообщения (подписка оформлена, пополнение Crypto Pay
# зачислено). NOTIFY_INFO=0 отключает их; заявки СБП и обращения в
# поддержку приходят всегда.
NOTIFY_INFO = os.getenv("NOTIFY_INFO", "1").strip() != "0"

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

# Pro (LongCat-Video-Avatar) — премиум-тариф: цена для клиента фиксированная,
# БЕЗ наценки поверх RUNPOD_COST_PER_SECOND_USD (та относится к Basic).
# $0.25 за секунду аудио. Если нужно поменять цену — меняется только это число
# (и такое же число PRO_PRICE_PER_SECOND_USD во фронтенде, src/api.js —
# оно используется только для текста-подсказки в интерфейсе).
PRO_PRICE_PER_SECOND_USD = 0.25
# Минимум секунд к оплате для Pro (по умолчанию 0 — отключено). Короткие ролики
# невыгодны: GPU в любом случае считает минимум один сегмент (~3.7 с) и
# загружает модель. Включается переменной окружения на Railway, например 4.
PRO_MIN_BILLABLE_SECONDS = float(os.getenv("PRO_MIN_BILLABLE_SECONDS", "0"))

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


class NotifyTestBody(BaseModel):
    admin_secret: str


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


def credit_deposit_if_pending(db: Session, invoice: dict):
    """Общая логика зачисления по оплаченному инвойсу Crypto Pay —
    используется и из вебхука (см. /crypto-webhook), и из ручного
    опроса /check-deposit из браузера. Идемпотентна: если депозит уже
    не в статусе pending (например, оба пути сработали почти
    одновременно), просто ничего не делает — защита от двойного
    зачисления. .with_for_update() блокирует строку на время
    транзакции, чтобы исключить гонку между вебхуком и опросом.

    Возвращает словарь с данными зачисления (user_id, method, amount,
    amount_usd), если зачисление выполнено ИМЕННО этим вызовом, иначе
    False. По этому результату вызывающий код шлёт информационное
    уведомление: оно приходит ровно один раз."""
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
    # данные для уведомления берём ДО commit (после него объекты обновляются из БД)
    info = {
        "user_id": deposit.user_id,
        "method": deposit.method,
        "amount": deposit.amount,
        "amount_usd": amount_usd,
    }
    db.commit()
    return info


def verify_crypto_pay_signature(raw_body: bytes, signature_header: str) -> bool:
    """Проверка подлинности вебхука — см. документацию Crypto Pay:
    подпись — это hex HMAC-SHA256 от тела запроса (как есть, без
    парсинга), ключ — SHA256-хэш токена приложения."""
    if not CRYPTO_PAY_TOKEN or not signature_header:
        return False
    secret_key = hashlib.sha256(CRYPTO_PAY_TOKEN.encode()).digest()
    computed = hmac.new(secret_key, raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature_header)


def send_telegram_message(chat_id: str, text: str) -> bool:
    """ВАЖНО: это синхронная блокирующая функция (requests.post). Она
    ДОЛЖНА вызываться только через asyncio.to_thread() из async-кода —
    иначе, пока Telegram API отвечает (вплоть до timeout=10 секунд),
    единственный поток event loop FastAPI полностью замирает, и сервер
    перестаёт отвечать вообще на ВСЕ запросы от ВСЕХ пользователей, а
    не только на этот один (тот же класс бага, что раньше уже был
    найден и исправлен в runpod_avatar.py для вызовов RunPod — здесь
    он был пропущен). См. историю чата: именно это вызвало 10-минутное
    зависание заявки СБП и одновременный сбой у другого пользователя,
    никак не связанного с СБП.

    Возвращает True, если Telegram принял сообщение."""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        print("TELEGRAM_BOT_TOKEN/ADMIN_TELEGRAM_CHAT_ID не настроены — уведомление не отправлено")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=10,
        )
        return r.ok
    except requests.exceptions.RequestException as e:
        print(f"Не удалось отправить Telegram-уведомление: {e}")
        return False


def send_max_message(text: str, button_text: str = None, button_url: str = None) -> tuple:
    """Отправка сообщения админу в MAX (POST {MAX_API_URL}/messages).
    Токен передаётся в заголовке Authorization без слова Bearer (так
    требует API MAX). Если задана кнопка, под сообщением показывается
    кнопка-ссылка (inline_keyboard, type=link).

    Как и send_telegram_message(), это СИНХРОННАЯ блокирующая функция:
    вызывать только через asyncio.to_thread().

    Возвращает (успех: bool, пояснение: str)."""
    if not MAX_API_TOKEN:
        print("MAX_API_TOKEN не настроен — уведомление в MAX не отправлено")
        return False, "MAX_API_TOKEN не настроен"
    if MAX_ADMIN_CHAT_ID:
        params = {"chat_id": MAX_ADMIN_CHAT_ID}
    elif MAX_ADMIN_USER_ID:
        params = {"user_id": MAX_ADMIN_USER_ID}
    else:
        return False, "не задан MAX_ADMIN_USER_ID / MAX_ADMIN_CHAT_ID"

    body = {"text": text[:4000]}
    if button_text and button_url:
        body["attachments"] = [{
            "type": "inline_keyboard",
            "payload": {"buttons": [[{"type": "link", "text": button_text, "url": button_url}]]},
        }]
    headers = {"Authorization": MAX_API_TOKEN, "Content-Type": "application/json"}
    url = f"{MAX_API_URL}/messages"

    try:
        try:
            r = requests.post(url, params=params, headers=headers, json=body, timeout=10)
        except requests.exceptions.SSLError as e:
            # Бот bestconsulting-AI ходит в MAX с verify=False: у площадки бывают
            # проблемы с цепочкой сертификатов. Повторяем один раз так же.
            print(f"MAX: ошибка проверки сертификата ({e}); повтор без проверки")
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            r = requests.post(url, params=params, headers=headers, json=body, timeout=10, verify=False)
        if r.status_code == 200:
            return True, "ok"
        info = f"{r.status_code}: {r.text[:200]}"
        print(f"MAX: уведомление не отправлено — {info}")
        return False, info
    except requests.exceptions.RequestException as e:
        print(f"Не удалось отправить MAX-уведомление: {e}")
        return False, str(e)[:200]


def notify_admin(text: str, button_text: str = None, button_url: str = None) -> bool:
    """Уведомление админу во все включённые каналы (NOTIFY_CHANNELS).
    Сбой одного канала не мешает другому. Синхронная — вызывать через
    asyncio.to_thread(). Возвращает True, если сообщение дошло хотя бы
    в один канал."""
    sent = False
    if "telegram" in NOTIFY_CHANNELS:
        try:
            sent = bool(send_telegram_message(ADMIN_TELEGRAM_CHAT_ID, text)) or sent
        except Exception as e:
            print(f"Telegram-уведомление: непредвиденная ошибка {e}")
    if "max" in NOTIFY_CHANNELS:
        try:
            ok, _info = send_max_message(text, button_text, button_url)
            sent = ok or sent
        except Exception as e:
            print(f"MAX-уведомление: непредвиденная ошибка {e}")
    return sent


def notify_info(text: str) -> bool:
    """Информационное сообщение админу (без кнопки). Отключается
    переменной NOTIFY_INFO=0."""
    if not NOTIFY_INFO:
        return False
    return notify_admin(text)


def notify_configured() -> bool:
    """Есть ли хотя бы один включённый и настроенный канал уведомлений
    (используется чатом поддержки перед отправкой обращения)."""
    tg = "telegram" in NOTIFY_CHANNELS and bool(TELEGRAM_BOT_TOKEN and ADMIN_TELEGRAM_CHAT_ID)
    mx = "max" in NOTIFY_CHANNELS and bool(MAX_API_TOKEN and (MAX_ADMIN_CHAT_ID or MAX_ADMIN_USER_ID))
    return tg or mx


# Ссылки на фоновые задачи уведомлений: без них Python может удалить задачу
# до её завершения.
_bg_tasks = set()


def _fire_and_forget(func, *args):
    """Запускает синхронную функцию в отдельном потоке и НЕ ждёт её
    завершения: клиент получает ответ сразу, даже если Telegram или MAX
    отвечают медленно."""
    task = asyncio.create_task(asyncio.to_thread(func, *args))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def make_sbp_confirm_token(deposit_id: int) -> str:
    """Короткий токен, привязанный к конкретной заявке (не универсальный
    пароль) — вычисляется из ADMIN_SECRET, так что ссылку можно смело
    отправлять в Telegram/MAX: она работает только для этой одной заявки,
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
    notify_admin(text, "Подтвердить зачисление", confirm_url)


def deposit_info_text(c: dict) -> str:
    """Текст информационного сообщения о зачисленном пополнении Crypto Pay."""
    return (
        "\U0001F4B5 Пополнение баланса (Crypto Pay)\n"
        f"Клиент: {c['user_id']}\n"
        f"Оплачено: {c['amount']} {c['method']}\n"
        f"Зачислено: ${c['amount_usd']:.2f}"
    )


def calculate_generation_cost(duration_seconds: float, tier: str = "basic") -> float:
    if tier == "pro":
        billable = max(duration_seconds, PRO_MIN_BILLABLE_SECONDS)
        return round(billable * PRO_PRICE_PER_SECOND_USD, 4)
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
        credited = credit_deposit_if_pending(db, invoice)
        if credited:
            _fire_and_forget(notify_info, deposit_info_text(credited))

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
        credited = credit_deposit_if_pending(db, invoice)
        if credited:
            _fire_and_forget(notify_info, deposit_info_text(credited))

    return {"ok": True}


@router.post("/sbp/request")
async def sbp_request(
    req: SbpRequestBody,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    """Клиент выбрал способ оплаты СБП -> создаётся заявка 'pending' ->
    фронтенд показывает номер телефона (SBP_PHONE). Админу мгновенно
    приходит уведомление в Telegram и/или MAX (NOTIFY_CHANNELS) со
    ссылкой на подтверждение в один клик (см. /sbp/confirm-link) — без
    Swagger/curl. Заявка общая для Avatar Studio и Video Studio:
    у них единый баланс.

    Уведомление уходит в отдельном потоке (to_thread) и в фоне: см.
    подробный комментарий у send_telegram_message() про то, почему
    синхронный requests.post() здесь НЕЛЬЗЯ вызывать напрямую, а фон
    нужен, чтобы клиент не ждал медленный ответ мессенджеров."""
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

    _fire_and_forget(notify_admin_new_sbp_request, deposit.id, req.amount_rub, user_id)

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


@router.post("/notify-test")
async def notify_test(req: NotifyTestBody):
    """Проверка уведомлений без реального платежа: отправляет тестовое
    сообщение во все включённые каналы и показывает результат по каждому.
    Защищено ADMIN_SECRET. Удобно вызывать из Swagger (/docs)."""
    if not ADMIN_SECRET or not hmac.compare_digest(req.admin_secret.encode(), ADMIN_SECRET.encode()):
        raise HTTPException(status_code=403, detail="Неверный admin_secret")

    text = "\u2705 Тест уведомлений BestConsulting (Avatar Studio / Video Studio). Если вы это видите, канал настроен."
    result = {}
    if "telegram" in NOTIFY_CHANNELS:
        result["telegram"] = {"ok": await asyncio.to_thread(send_telegram_message, ADMIN_TELEGRAM_CHAT_ID, text)}
    if "max" in NOTIFY_CHANNELS:
        ok, info = await asyncio.to_thread(send_max_message, text, "Открыть бэкенд", BACKEND_PUBLIC_URL)
        result["max"] = {"ok": ok, "info": info}
    return {"channels": sorted(NOTIFY_CHANNELS), "info_messages": NOTIFY_INFO, "result": result}


@router.get("/sbp/confirm-link", response_class=HTMLResponse)
async def sbp_confirm_link(deposit_id: int, token: str, db: Session = Depends(get_db)):
    """Ссылка из уведомления (Telegram/MAX) — одна ссылка подтверждает
    ровно одну заявку. Открывается прямо в браузере (в том числе с
    телефона, прямо из мессенджера), никакого Swagger/curl/ручного
    ввода ADMIN_SECRET не требуется."""
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
    случай, если уведомление не пришло/потерялось) — тот же механизм,
    что и раньше."""
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
async def estimate_cost(duration_seconds: float, tier: str = "basic"):
    cost = calculate_generation_cost(duration_seconds, tier)
    return {
        "duration_seconds": duration_seconds,
        "tier": tier,
        "estimated_cost_usd": cost,
    }


# ---------- Единый баланс: списания за платные функции Video Studio ----------

def charge_user(
    db: Session,
    user_id: str,
    amount_usd: float,
    feature: str,
    units: float = None,
    note: str = None,
    product: str = "video-studio",
) -> float:
    """ВНУТРЕННЯЯ функция списания — это НЕ роут, из браузера её вызвать
    нельзя. Вызывается только из серверных эндпоинтов, которые сами
    выполняют платную работу (синтез речи, удаление голоса, облачный
    рендер) и сами считают цену на сервере. Так клиент не может
    подставить свою цену.

    Списание атомарное: строка пользователя блокируется на время
    транзакции (with_for_update), поэтому два одновременных запроса не
    смогут уйти в минус. При нехватке средств — HTTP 402. Возвращает
    новый баланс."""
    amount = round(float(amount_usd), 4)
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Некорректная сумма списания")

    user = db.query(User).filter(User.user_id == user_id).with_for_update().first()
    if not user or (user.balance_usd or 0.0) < amount:
        db.rollback()
        raise HTTPException(status_code=402, detail="Недостаточно средств на балансе")

    user.balance_usd -= amount
    db.add(Charge(
        user_id=user_id,
        product=product,
        feature=feature,
        amount_usd=amount,
        units=units,
        note=note,
    ))
    db.commit()
    return round(user.balance_usd, 4)


@router.get("/history")
async def billing_history(db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Последние операции пользователя: подтверждённые пополнения и
    списания за функции Video Studio (списания за генерации Avatar
    Studio в этот журнал пока не пишутся)."""
    deposits = (
        db.query(Deposit)
        .filter(Deposit.user_id == user_id, Deposit.status == "confirmed")
        .order_by(Deposit.created_at.desc())
        .limit(30)
        .all()
    )
    charges = (
        db.query(Charge)
        .filter(Charge.user_id == user_id)
        .order_by(Charge.created_at.desc())
        .limit(30)
        .all()
    )
    items = [
        {
            "kind": "deposit",
            "amount_usd": round(d.amount_usd or 0.0, 4),
            "label": f"Пополнение · {d.method}",
            "at": (d.confirmed_at or d.created_at).isoformat() + "Z",
        }
        for d in deposits
    ]
    items += [
        {
            "kind": "charge",
            "amount_usd": -round(c.amount_usd, 4),
            "label": f"{c.product} · {c.feature}",
            "at": c.created_at.isoformat() + "Z",
        }
        for c in charges
    ]
    items.sort(key=lambda x: x["at"], reverse=True)
    return {"items": items[:30]}


# ---------- Подписка Video Studio: 199 ₽ / 30 дней, оплата с общего баланса ----------

VIDEO_SUB_PRODUCT = "video-studio"
VIDEO_SUB_PRICE_RUB = 199.0
VIDEO_SUB_DAYS = 30


def video_sub_price_usd() -> float:
    """Цена подписки в USD по тому же курсу, что и рублёвые пополнения.
    Округление ВНИЗ до 4 знаков: пополнение ровно на 199 ₽ (по СБП или
    через Crypto Pay в RUB) зачисляется как 199 * RUB_TO_USD_RATE и
    всегда покрывает цену подписки без копеечной недостачи."""
    return math.floor(VIDEO_SUB_PRICE_RUB * RUB_TO_USD_RATE * 10000) / 10000


def _sub_payload(sub) -> dict:
    now = datetime.utcnow()
    active = bool(sub and sub.until and sub.until > now)
    return {
        "active": active,
        "until": (sub.until.isoformat() + "Z") if sub and sub.until else None,
        "price_rub": VIDEO_SUB_PRICE_RUB,
        "price_usd": video_sub_price_usd(),
        "days": VIDEO_SUB_DAYS,
    }


def subscription_info_text(user_id: str, renewed: bool, until: datetime, price_usd: float, balance_usd: float) -> str:
    """Текст информационного сообщения о подписке Video Studio."""
    return (
        "\U0001F514 Подписка Video Studio " + ("продлена" if renewed else "оформлена") + "\n"
        f"Клиент: {user_id}\n"
        f"Списано: {VIDEO_SUB_PRICE_RUB:.0f} \u20bd (${price_usd:.2f}) с баланса\n"
        f"Действует до: {until:%d.%m.%Y %H:%M} UTC\n"
        f"Остаток баланса: ${balance_usd:.2f}"
    )


@router.get("/subscription")
async def get_subscription(db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Статус подписки Video Studio у текущего пользователя + цена."""
    sub = db.query(Subscription).filter(
        Subscription.user_id == user_id,
        Subscription.product == VIDEO_SUB_PRODUCT,
    ).first()
    return _sub_payload(sub)


@router.post("/subscribe")
async def subscribe_video(db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Покупка/продление подписки Video Studio с общего баланса.
    Списание, запись в журнал и продление подписки — в ОДНОЙ транзакции:
    либо всё, либо ничего. Цену считает сервер, клиент её не передаёт.
    Если подписка ещё действует, +30 дней добавляются к её концу.
    После успешной покупки админу уходит информационное сообщение
    (Telegram и/или MAX), в фоне, чтобы не задерживать ответ клиенту."""
    price = video_sub_price_usd()
    now = datetime.utcnow()

    user = db.query(User).filter(User.user_id == user_id).with_for_update().first()
    if not user or (user.balance_usd or 0.0) < price:
        db.rollback()
        raise HTTPException(status_code=402, detail="Недостаточно средств на балансе")

    # защита от двойного клика: вторая покупка в течение 15 секунд отклоняется
    recent = db.query(Charge).filter(
        Charge.user_id == user_id,
        Charge.feature == "subscription",
        Charge.created_at > now - timedelta(seconds=15),
    ).first()
    if recent:
        db.rollback()
        raise HTTPException(status_code=409, detail="Подписка уже оформляется, подождите несколько секунд")

    sub = db.query(Subscription).filter(
        Subscription.user_id == user_id,
        Subscription.product == VIDEO_SUB_PRODUCT,
    ).with_for_update().first()
    was_active = bool(sub and sub.until and sub.until > now)
    start = sub.until if was_active else now
    until = start + timedelta(days=VIDEO_SUB_DAYS)
    if sub:
        sub.until = until
    else:
        sub = Subscription(user_id=user_id, product=VIDEO_SUB_PRODUCT, until=until)
        db.add(sub)

    user.balance_usd -= price
    db.add(Charge(
        user_id=user_id,
        product=VIDEO_SUB_PRODUCT,
        feature="subscription",
        amount_usd=price,
        units=VIDEO_SUB_DAYS,
        note=f"{VIDEO_SUB_PRICE_RUB:.0f} ₽ / {VIDEO_SUB_DAYS} дн.",
    ))
    db.commit()

    payload = _sub_payload(sub)
    payload["balance_usd"] = round(user.balance_usd, 4)

    _fire_and_forget(
        notify_info,
        subscription_info_text(user_id, was_active, until, price, payload["balance_usd"]),
    )
    return payload
