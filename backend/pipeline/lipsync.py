"""
Модуль синхронизации губ с речью.

Ключевая инженерная идея для стабильности лица на роликах до 5 минут:
НЕ генерировать 5 минут одним проходом. Модели липсинка (MuseTalk и
аналоги) стабильны на коротких окнах (~15-30 сек) и накапливают дрейф
на длинных. Поэтому сегмент режется на чанки по CHUNK_SECONDS, каждый
чанк генерируется НЕЗАВИСИМО от опорных кадров аватара (avatar.py),
а не от последнего кадра предыдущего чанка — это и есть защита от
"расползания" лица. Чанки затем склеиваются встык (без кроссфейда
внутри одного сегмента, т.к. это один непрерывный кусок речи).

Итоговая склейка НЕСКОЛЬКИХ сегментов в фильм — это отдельный шаг,
см. stitch.py, там уже осмысленный кроссфейд между разными сценами.

Реальный инференс идёт через MuseTalk CLI (scripts/inference.py) —
именно так пайплайн был проверен вручную и подтверждён рабочим на
этой машине (см. results/test/v15/*.mp4).

Создание задачи (create_job) и её выполнение (run_job) разделены
намеренно: create_job — быстрая операция (доли секунды, только ffprobe),
она вызывается прямо в обработчике HTTP-запроса. run_job — медленная
(минуты), она выполняется в фоновом потоке, чтобы не держать HTTP-
соединение открытым на всё время генерации (см. main.py).

Эмоции (v2): вместо статичного опорного кадра аватара как видео-источника
для MuseTalk можно подставить один из готовых видео-драйверов эмоций
(data/emotion_drivers/*.mp4) — короткие ролики с естественной мимикой.
MuseTalk в этом случае переносит липсинк поверх движения из драйвера,
а не поверх неподвижного фото. Если драйвер для выбранной эмоции
отсутствует на диске — используется прежнее поведение (fallback на
reference_frames[0] аватара), чтобы функциональность не ломалась,
пока не все ролики записаны (например, love.mp4).
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

CHUNK_SECONDS = 20  # эмпирический предел стабильности для MuseTalk-класса моделей
MAX_SEGMENT_SECONDS = 5 * 60

# backend/pipeline/lipsync.py -> backend/ -> backend/MuseTalk
MUSETALK_DIR = Path(__file__).resolve().parent.parent / "MuseTalk"
UNET_MODEL_PATH = MUSETALK_DIR / "models" / "musetalkV15" / "unet.pth"
UNET_CONFIG_PATH = MUSETALK_DIR / "models" / "musetalkV15" / "musetalk.json"
MUSETALK_VERSION = "v15"

# backend/pipeline/lipsync.py -> backend/ -> backend/data/emotion_drivers
EMOTION_DRIVERS_DIR = Path(__file__).resolve().parent.parent / "data" / "emotion_drivers"

# Соответствие пресета эмоции файлу-драйверу. "love" оставлен в списке
# намеренно — как только ролик будет записан и положен в эту папку,
# он подхватится автоматически, без правок кода.
EMOTION_DRIVER_FILES: dict[str, str] = {
    "neutral": "neutral.mp4",
    "happy": "happy.mp4",
    "sadness": "sadness.mp4",
    "anger": "anger.mp4",
    "love": "love.mp4",
}


def list_available_emotions() -> list[dict]:
    """
    Возвращает список пресетов эмоций с флагом, реально ли для них есть
    файл-драйвер на диске сейчас. Используется эндпоинтом GET /emotions,
    чтобы фронтенд не предлагал пользователю ещё не записанные эмоции.
    """
    result = []
    for emotion, filename in EMOTION_DRIVER_FILES.items():
        path = EMOTION_DRIVERS_DIR / filename
        result.append({
            "id": emotion,
            "available": path.exists() and path.stat().st_size > 0,
        })
    return result


def resolve_video_source(emotion: str, avatar_reference_frame: str) -> str:
    """
    Выбирает, что подать в MuseTalk как video_path для данного сегмента:
    - если для emotion есть существующий ненулевой файл-драйвер —
      возвращаем его (естественное движение вместо статичного фото);
    - иначе — прежнее поведение: статичный опорный кадр аватара.
    """
    filename = EMOTION_DRIVER_FILES.get(emotion)
    if filename:
        driver_path = EMOTION_DRIVERS_DIR / filename
        if driver_path.exists() and driver_path.stat().st_size > 0:
            return str(driver_path)
    return avatar_reference_frame


@dataclass
class LipsyncJob:
    job_id: str
    avatar_id: str
    audio_path: str
    total_duration_sec: float
    chunk_count: int
    emotion: str = "neutral"
    status: str = "pending"  # pending | running | done | failed
    output_path: str | None = None
    error: str | None = None


def plan_chunks(total_duration_sec: float) -> list[tuple[float, float]]:
    """Возвращает список (start, end) в секундах для чанкованной генерации."""
    if total_duration_sec > MAX_SEGMENT_SECONDS:
        raise ValueError(
            f"Сегмент {total_duration_sec:.0f} сек превышает лимит "
            f"{MAX_SEGMENT_SECONDS} сек (5 минут) для одного ролика. "
            f"Разбейте на несколько сегментов и соберите их через stitch.py."
        )
    n_chunks = math.ceil(total_duration_sec / CHUNK_SECONDS)
    chunks = []
    for i in range(n_chunks):
        start = i * CHUNK_SECONDS
        end = min((i + 1) * CHUNK_SECONDS, total_duration_sec)
        chunks.append((start, end))
    return chunks


def _slice_audio(audio_path: str, start: float, end: float, out_path: Path) -> Path:
    """Вырезает кусок аудио [start, end) секунд через ffmpeg, перекодируя
    в чистый WAV — MuseTalk использует Whisper-фичи, лишние проблемы с
    контейнером на входе лучше исключить сразу."""
    cmd = [
        "ffmpeg", "-y", "-v", "warning",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", audio_path,
        "-ac", "1", "-ar", "24000",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg (нарезка аудио) failed: {result.stderr}")
    return out_path


def _run_musetalk_chunk(video_source: str, chunk_audio: Path, work_dir: Path, task_name: str) -> Path:
    """
    Запускает один чанк через реальный MuseTalk CLI — тот же способ,
    который был проверен вручную (scripts.inference с YAML-конфигом).

    video_source может быть как статичным опорным кадром аватара (фото),
    так и видео-драйвером эмоции (см. resolve_video_source) — MuseTalk
    принимает оба варианта одинаково через video_path в конфиге.
    """
    config_path = work_dir / f"{task_name}.yaml"
    config_path.write_text(
        f'{task_name}:\n'
        f'  video_path: "{Path(video_source).resolve().as_posix()}"\n'
        f'  audio_path: "{chunk_audio.resolve().as_posix()}"\n',
        encoding="utf-8",
    )

    result_dir = work_dir / "result"
    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config", str(config_path.resolve()),
        "--unet_model_path", str(UNET_MODEL_PATH.resolve()),
        "--unet_config", str(UNET_CONFIG_PATH.resolve()),
        "--version", MUSETALK_VERSION,
        "--result_dir", str(result_dir.resolve()),
    ]

    # Windows по умолчанию перехватывает stdout/stderr дочернего процесса в
    # системной кодировке консоли (обычно cp1251), а MuseTalk иногда печатает
    # символы вне этой кодовой страницы (например, юникодные кавычки в
    # подсказках) — без принудительного UTF-8 процесс падает с
    # UnicodeEncodeError раньше, чем успевает сохранить готовое видео.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    log_path = work_dir / f"{task_name}.log"
    result = subprocess.run(
        cmd, cwd=str(MUSETALK_DIR), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    log_path.write_text(
        f"CMD: {' '.join(cmd)}\n\n"
        f"--- STDOUT ---\n{result.stdout}\n\n"
        f"--- STDERR ---\n{result.stderr}\n",
        encoding="utf-8", errors="replace",
    )

    # MuseTalk называет выходной файл по схеме {video_stem}_{audio_stem}.mp4.
    # Важно: код возврата процесса НЕ проверяем как первый признак успеха —
    # у MuseTalk бывает необязательный шаг после сборки видео (сохранение
    # промежуточных кадров), который может упасть с ошибкой уже ПОСЛЕ того,
    # как итоговый .mp4 реально создан и валиден. Поэтому решающий критерий —
    # существование и ненулевой размер выходного файла, а не returncode.
    # Дополнительно даём файловой системе немного времени: ffmpeg внутри
    # MuseTalk иногда возвращает управление на миллисекунды раньше, чем
    # файл гарантированно долетает до диска на Windows.
    video_stem = Path(video_source).stem
    audio_stem = chunk_audio.stem
    output = result_dir / MUSETALK_VERSION / f"{video_stem}_{audio_stem}.mp4"

    for _ in range(10):
        if output.exists() and output.stat().st_size > 0:
            break
        time.sleep(0.5)

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(
            f"MuseTalk inference failed для чанка {task_name} "
            f"(returncode={result.returncode}). Полный лог: {log_path}\n"
            f"Последние строки:\n{result.stderr[-2000:]}"
        )

    return output


def create_job(avatar_id: str, audio_path: str, emotion: str = "neutral") -> LipsyncJob:
    """
    Создаёт объект задачи и сразу считает длительность/чанки (это быстрая
    операция через ffprobe, доли секунды) — БЕЗ запуска самого инференса.
    Возвращается мгновенно, чтобы HTTP-ответ не держал соединение открытым
    на все время генерации — саму генерацию делает run_job() в фоне,
    а фронтенд узнаёт результат через короткие опросы /jobs/{id}.
    """
    from .voice_clone import _probe_duration_seconds

    duration = _probe_duration_seconds(audio_path)
    chunks = plan_chunks(duration)

    return LipsyncJob(
        job_id=str(uuid.uuid4())[:8],
        avatar_id=avatar_id,
        audio_path=audio_path,
        total_duration_sec=duration,
        chunk_count=len(chunks),
        emotion=emotion,
        status="pending",
    )


def run_job(job: LipsyncJob) -> None:
    """
    Выполняет реальную (медленную) генерацию сегмента: режет аудио на
    чанки, для каждого прогоняет липсинк-модель, склеивает результат.
    Мутирует переданный job-объект на месте по мере прогресса — вызывающий
    код (main.py) хранит этот же объект в JOBS, поэтому GET /jobs/{id}
    сразу видит актуальный статус без дополнительной синхронизации.

    Рассчитана на вызов в фоновом потоке (не в основном asyncio-цикле
    FastAPI), т.к. сам инференс — блокирующий CPU-bound процесс.
    """
    from . import avatar as avatar_module
    from .stitch import concat_video_chunks

    job.status = "running"

    try:
        profile = avatar_module.load_avatar_profile(job.avatar_id)
        if not profile.reference_frames:
            raise RuntimeError(f"У аватара {job.avatar_id} нет опорных кадров")
        avatar_frame = profile.reference_frames[0]

        # Если для выбранной эмоции есть готовый видео-драйвер — используем
        # его как источник движения для всех чанков этого сегмента, иначе
        # fallback на прежнее поведение (статичный кадр аватара).
        video_source = resolve_video_source(job.emotion, avatar_frame)

        work_dir = Path("data") / "lipsync_jobs" / job.job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        chunks = plan_chunks(job.total_duration_sec)
        chunk_videos = []
        for i, (start, end) in enumerate(chunks):
            task_name = f"chunk_{i}"
            chunk_audio = _slice_audio(
                job.audio_path, start, end, work_dir / f"{task_name}.wav"
            )
            chunk_video = _run_musetalk_chunk(
                video_source, chunk_audio, work_dir, task_name
            )
            chunk_videos.append(str(chunk_video))

        output_path = str(work_dir / "segment.mp4")
        concat_video_chunks(chunk_videos, output_path)

        job.output_path = output_path
        job.status = "done"
    except Exception as e:
        job.status = "failed"
        job.error = str(e)