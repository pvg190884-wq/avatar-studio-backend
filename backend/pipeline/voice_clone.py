"""
Модуль клонирования голоса.

Используется XTTS-v2 (через пакет coqui-tts) — zero-shot клонирование:
достаточно 6-15 секунд чистой речи пользователя, дообучение не требуется,
голос "клонируется" на лету при каждом синтезе. Полностью локально,
без обращения к внешним API.

Лицензия: XTTS-v2 распространяется под Coqui Public Model License —
некоммерческое использование свободно, коммерческое требует отдельного
согласования с правообладателем модели. Проверьте актуальный статус
лицензии перед стартом монетизации приложения.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

# XTTS-v2 при первом запуске в интерактивном режиме спрашивает согласие
# с некоммерческой лицензией (y/n в консоли). Наш бэкенд работает как
# фоновый процесс без интерактивного ввода — без этой переменной первый
# синтез молча зависнет в ожидании ответа, которого никто не даст.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

VOICES_DIR = Path("data/voices")
MIN_SAMPLE_SECONDS = 6
RECOMMENDED_SAMPLE_SECONDS = 15

# Модель грузится один раз на процесс и держится в памяти — сама загрузка
# весов (~1.8ГБ) с диска занимает заметное время, повторять это на каждый
# синтез бессмысленно и на CPU особенно ощутимо.
_tts_engine = None


def _get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        import torch
        from TTS.api import TTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _tts_engine = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _tts_engine


@dataclass
class VoiceProfile:
    voice_id: str
    name: str
    sample_path: str
    sample_duration_sec: float
    consent_confirmed: bool

    def to_json(self) -> dict:
        return asdict(self)


def clone_voice(
    sample_audio_path: str,
    name: str,
    consent_confirmed: bool,
) -> VoiceProfile:
    """
    Регистрирует голосовой профиль по образцу речи пользователя.
    Само "клонирование" в XTTS-v2 происходит не на этом шаге, а на шаге
    synthesize() — модель каждый раз берёт сэмпл заново как референс тембра.
    Здесь мы только валидируем и сохраняем сэмпл.
    """
    if not consent_confirmed:
        raise ValueError(
            "Клонирование голоса требует явного согласия "
            "(consent_confirmed=True) — это голос конкретного человека."
        )

    duration = _probe_duration_seconds(sample_audio_path)
    if duration < MIN_SAMPLE_SECONDS:
        raise ValueError(
            f"Образец слишком короткий ({duration:.1f} сек). "
            f"Нужно минимум {MIN_SAMPLE_SECONDS} сек, "
            f"рекомендуется {RECOMMENDED_SAMPLE_SECONDS}+ сек чистой речи "
            f"без музыки и шума на фоне."
        )

    voice_id = str(uuid.uuid4())[:8]
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)

    source = Path(sample_audio_path)
    stored_sample = voice_dir / f"sample{source.suffix}"
    shutil.copy(source, stored_sample)

    profile = VoiceProfile(
        voice_id=voice_id,
        name=name,
        sample_path=str(stored_sample),
        sample_duration_sec=duration,
        consent_confirmed=True,
    )
    (voice_dir / "profile.json").write_text(
        json.dumps(profile.to_json(), ensure_ascii=False, indent=2)
    )
    return profile


def synthesize(text: str, voice_id: str, language: str = "ru") -> Path:
    """
    Синтезирует речь заданным клонированным голосом через XTTS-v2.
    Первый вызов в процессе будет заметно медленнее последующих — тогда
    же скачиваются веса модели (~1.8ГБ), если их ещё нет на диске.
    """
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise FileNotFoundError(f"Голосовой профиль {voice_id} не найден")

    sample_files = list(voice_dir.glob("sample.*"))
    if not sample_files:
        raise FileNotFoundError(
            f"Аудио-образец для голоса {voice_id} не найден на диске"
        )
    speaker_wav = str(sample_files[0])

    output_dir = voice_dir / "generated"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4().hex[:8]}.wav"

    tts = _get_tts_engine()
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=str(output_path),
    )
    return output_path


def _probe_duration_seconds(audio_path: str) -> float:
    """Определяет длительность аудио через ffprobe (входит в состав ffmpeg)."""
    import subprocess

    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())
