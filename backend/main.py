"""
Avatar Studio — облачный бэкенд (прокси к RunPod Serverless).
Разворачивается на Railway, работает 24/7 независимо от локальной машины.
Хранит RunPod API-ключ на сервере, принимает запросы от сайта, передаёт
их в RunPod, возвращает готовое видео.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.runpod_avatar import router as runpod_avatar_router
from routers.billing import router as billing_router

app = FastAPI(title="Avatar Studio Backend (Cloud)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # сузим до конкретного домена сайта, когда он будет готов
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(runpod_avatar_router)
app.include_router(billing_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
