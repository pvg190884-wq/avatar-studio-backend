"""
Avatar Studio — подключение к базе данных.

Использует Postgres (Railway plugin) через переменную окружения DATABASE_URL,
которую Railway создаёт автоматически при добавлении Postgres-плагина к
проекту — вручную её вписывать не нужно.
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_dev.db")
# Railway иногда отдаёт URL со схемой postgres:// вместо postgresql://,
# которую современный SQLAlchemy уже не принимает — приводим к нужному виду.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)  # telegram id / email / телефон
    balance_usd = Column(Float, default=0.0)


class Deposit(Base):
    """Единая таблица для всех пополнений — и крипто (Crypto Pay), и СБП
    (ручное подтверждение). Поле method хранит способ ("RUB", "USD", "BTC",
    "TON", "USDT", "SBP"), status — "pending" / "confirmed" / "rejected"."""
    __tablename__ = "deposits"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    method = Column(String, nullable=False)
    amount = Column(Float, nullable=False)          # сумма в исходной валюте (как ввёл клиент)
    amount_usd = Column(Float, nullable=True)        # сумма, зачисленная на баланс в USD (после конвертации)
    status = Column(String, default="pending")
    external_id = Column(String, nullable=True)      # invoice_id от Crypto Pay, если применимо
    note = Column(String, nullable=True)             # для СБП — что клиент указал как подтверждение (номер операции)
    created_at = Column(DateTime, default=datetime.utcnow)
    confirmed_at = Column(DateTime, nullable=True)


class Charge(Base):
    """Журнал списаний за платные функции Video Studio (и других продуктов
    на общем балансе). Заполняется только функцией charge_user() из
    billing.py — на стороне сервера, цена всегда считается сервером."""
    __tablename__ = "charges"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    product = Column(String, nullable=False)         # например "video-studio"
    feature = Column(String, nullable=False)         # "tts", "voice_removal", "cloud_render" и т.п.
    amount_usd = Column(Float, nullable=False)       # сколько списано
    units = Column(Float, nullable=True)             # за что: секунды, минуты, символы
    note = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
