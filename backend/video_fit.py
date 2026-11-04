"""
Сжатие видео под лимит RunPod: тело запроса /run не может быть больше 10 МиБ,
а видео и звук отправляются внутри запроса в виде base64 (на треть больше самого файла).

fit_video_for_runpod(video_path, audio_path):
  - если видео и так помещается вместе со звуком, возвращает тот же путь (без перекодирования);
  - иначе создаёт рядом сжатую копию (video_path + ".small.mp4") и возвращает её путь.
    Удалить эту копию должен вызывающий код.

СИНХРОННАЯ функция (запускает ffmpeg и ждёт): вызывать только через asyncio.to_thread.

Переменные окружения (необязательные):
  RUNPOD_BODY_BUDGET_BYTES — сколько байт запроса мы считаем допустимым (по умолчанию 9 МиБ: запас под обёртку JSON)
  VIDEO_COMPRESS_TIMEOUT   — предельное время одного прохода ffmpeg, секунд (по умолчанию 240)
"""
import os
import re
import math
import shutil
import subprocess

from fastapi import HTTPException

BODY_BUDGET_BYTES = int(os.getenv("RUNPOD_BODY_BUDGET_BYTES", str(9 * 1024 * 1024)))
ENCODE_TIMEOUT = int(os.getenv("VIDEO_COMPRESS_TIMEOUT", "240"))
AUDIO_TRACK_BPS = 32000   # звук самого видео нужен только как запасной: озвучку присылаем отдельно
MIN_VIDEO_BPS = 150000    # ниже этого битрейта картинка разваливается: проще попросить ролик покороче
MAX_FPS = 25              # воркер липсинка всё равно приводит видео к 25 кадрам/с


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise HTTPException(status_code=500, detail="На сервере нет ffmpeg, поэтому сжать видео не получилось. Сообщите в поддержку.")


def _probe(exe: str, path: str):
    """Длительность и частота кадров по выводу ffmpeg -i (ffprobe в пакете нет)."""
    try:
        r = subprocess.run([exe, "-hide_banner", "-i", path], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None, None
    text = r.stderr or ""
    duration = None
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    fps = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
    if m:
        fps = float(m.group(1))
    return duration, fps


def _encode(exe: str, src: str, dst: str, short_side: int, video_bps: int, fps):
    # короткая сторона не больше short_side (без увеличения): 1080x1920 -> 720x1280, 1920x1080 -> 1280x720
    vf = [f"scale=w='if(gte(iw,ih),-2,min({short_side},iw))':h='if(gte(iw,ih),min({short_side},ih),-2)'"]
    if fps and fps > MAX_FPS + 0.5:
        vf.append(f"fps={MAX_FPS}")
    cmd = [
        exe, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
        "-vf", ",".join(vf),
        "-c:v", "libx264", "-preset", "veryfast", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-b:v", str(video_bps), "-maxrate", str(int(video_bps * 1.5)), "-bufsize", str(int(video_bps * 3)),
        "-c:a", "aac", "-b:a", "32k", "-ac", "1",
        "-movflags", "+faststart",
        dst,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=ENCODE_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Не удалось сжать видео вовремя. Попробуйте ролик покороче (до 15 секунд).")
    if r.returncode != 0 or not os.path.exists(dst):
        tail = (r.stderr or "").strip().replace("\n", " ")[-300:]
        print(f"ffmpeg: ошибка сжатия видео: {tail}")
        raise HTTPException(status_code=422, detail="Не удалось обработать видео: файл повреждён или формат не поддерживается. Загрузите MP4 или MOV.")


def fit_video_for_runpod(video_path: str, audio_path: str) -> str:
    # сколько байт видео поместится в запрос после base64, если звук озвучки тоже едет в запросе
    audio_b64 = math.ceil(os.path.getsize(audio_path) / 3) * 4
    budget = (BODY_BUDGET_BYTES - audio_b64 - 8192) * 3 // 4
    if budget < 300_000:
        raise HTTPException(status_code=413, detail="Озвучка получилась слишком длинной для обработки. Сократите текст или аудио.")

    size = os.path.getsize(video_path)
    if size <= budget:
        return video_path

    exe = _ffmpeg_exe()
    duration, fps = _probe(exe, video_path)
    if not duration or duration <= 0:
        raise HTTPException(status_code=422, detail="Не удалось определить длительность видео. Загрузите MP4 или MOV.")

    total_bps = budget * 8 * 0.85 / duration  # 15% запас: битрейт ffmpeg выдерживает приблизительно
    video_bps = int(total_bps - AUDIO_TRACK_BPS)
    if video_bps < MIN_VIDEO_BPS:
        raise HTTPException(
            status_code=413,
            detail=f"Видео слишком длинное ({duration:.0f} с) для обработки. Сократите ролик до 15 секунд.",
        )

    # ступени разрешения: чем меньше битрейта, тем меньше кадр
    steps = [(720, 1.0), (540, 0.85), (480, 0.7)]
    start = 0 if video_bps >= 700_000 else (1 if video_bps >= 400_000 else 2)
    dst = video_path + ".small.mp4"
    try:
        for short_side, k in steps[start:]:
            _encode(exe, video_path, dst, short_side, int(video_bps * k), fps)
            out = os.path.getsize(dst)
            if out <= budget:
                print(f"Видео сжато под RunPod: {size / 1048576:.1f} МБ -> {out / 1048576:.1f} МБ ({short_side}p, {duration:.0f} с)")
                return dst
        raise HTTPException(
            status_code=413,
            detail="Видео слишком тяжёлое для обработки даже после сжатия. Сократите ролик до 10–15 секунд.",
        )
    except Exception:
        if os.path.exists(dst):
            os.remove(dst)
        raise
