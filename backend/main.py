"""
Avatar Studio — облачный бэкенд (прокси к RunPod Serverless).
Разворачивается на Railway, работает 24/7 независимо от локальной машины.
Хранит RunPod API-ключ на сервере, принимает запросы от сайта, передаёт
их в RunPod, возвращает готовое видео.
"""
import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routers.runpod_avatar import router as runpod_avatar_router
from routers.billing import router as billing_router
from database import init_db

# Необязательные разделы (ИИ и поддержка) подключаем с защитой: если файла нет или в нём ошибка,
# остальной бэкенд (генерации, баланс, подписка) всё равно запустится.
try:
    from routers.ai import router as ai_router
except Exception as e:
    ai_router = None
    print(f"ВНИМАНИЕ: раздел ИИ (routers/ai.py) не загружен: {e!r}")

try:
    from routers.support import router as support_router
except Exception as e:
    support_router = None
    print(f"ВНИМАНИЕ: раздел поддержки (routers/support.py) не загружен: {e!r}")

app = FastAPI(title="Avatar Studio Backend (Cloud)")


# ВАЖНО: этот перехватчик ошибок должен быть добавлен ДО CORS-прослойки (так он оказывается внутри неё).
# Иначе необработанная ошибка превращается в ответ без CORS-заголовков, и браузер показывает
# только «Failed to fetch» вместо настоящей причины.
@app.middleware("http")
async def catch_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print(f"Необработанная ошибка {request.method} {request.url.path}: {e!r}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Внутренняя ошибка сервера ({type(e).__name__}). Попробуйте ещё раз."},
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # сузим до конкретного домена сайта, когда он будет готов
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(runpod_avatar_router)
app.include_router(billing_router)
if ai_router is not None:
    app.include_router(ai_router)
if support_router is not None:
    app.include_router(support_router)


@app.on_event("startup")
async def on_startup():
    # Создаёт таблицы users/deposits при первом запуске, если их ещё нет.
    init_db()


@app.get("/health")
async def health():
    return {"status": "ok"}
