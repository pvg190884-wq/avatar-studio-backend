"""
Модуль аватаров: создание и хранение идентичности лица.

Идея стабильности: аватар — это не "картинка", а зафиксированный набор
эмбеддингов лица (InsightFace) + 3-5 опорных кадров с разными ракурсами,
снятых один раз при создании. Все последующие сегменты видео используют
ОДНИ И ТЕ ЖЕ опорные данные — это и есть главный приём против "расползания"
лица на длинных роликах: генерация никогда не отталкивается от предыдущего
сгенерированного кадра, только от зафиксированного оригинала.
"""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

AVATARS_DIR = Path("data/avatars")

# Модель InsightFace грузится один раз на процесс (это не быстрая операция),
# а не при каждом вызове — иначе каждое создание аватара заново инициализирует
# ONNX-раннтайм, что на CPU ощутимо медленнее, чем сам подсчёт эмбеддинга.
_face_analyzer = None


def _get_face_analyzer():
    global _face_analyzer
    if _face_analyzer is None:
        import insightface

        _face_analyzer = insightface.app.FaceAnalysis(name="buffalo_l")
        # ctx_id=-1 означает явный запуск на CPU — на этой машине нет
        # NVIDIA GPU, а insightface/onnxruntime сами не всегда падают
        # красиво, если не сказать им прямо, что CUDA использовать не нужно.
        _face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    return _face_analyzer


@dataclass
class AvatarProfile:
    avatar_id: str
    name: str
    consent_confirmed: bool
    reference_frames: list[str]  # пути к опорным кадрам лица
    embedding_path: str | None  # путь к сохранённым эмбеддингам лица (.npy)

    def to_json(self) -> dict:
        return asdict(self)


class AvatarNotReadyError(RuntimeError):
    pass


class NoFaceDetectedError(ValueError):
    """Явная, понятная пользователю ошибка вместо невнятного падения ONNX."""


def create_avatar_profile(
    source_media_path: str,
    name: str,
    consent_confirmed: bool,
) -> AvatarProfile:
    """
    Создаёт профиль аватара из согласного фото/видео пользователя.

    consent_confirmed обязателен: приложение не должно позволять создавать
    аватар из чужого лица без явного подтверждения, что это либо сам
    пользователь, либо у него есть право использовать эти данные.
    """
    if not consent_confirmed:
        raise ValueError(
            "Создание аватара требует явного подтверждения согласия "
            "(consent_confirmed=True). Это не техническое ограничение, "
            "а осознанное требование продукта."
        )

    avatar_id = str(uuid.uuid4())[:8]
    avatar_dir = AVATARS_DIR / avatar_id
    avatar_dir.mkdir(parents=True, exist_ok=True)

    source = Path(source_media_path)
    stored_source = avatar_dir / f"source{source.suffix}"
    shutil.copy(source, stored_source)

    reference_frames = _extract_reference_frames(stored_source, avatar_dir)
    embedding_path = _build_face_embedding(reference_frames, avatar_dir)

    profile = AvatarProfile(
        avatar_id=avatar_id,
        name=name,
        consent_confirmed=True,
        reference_frames=[str(p) for p in reference_frames],
        embedding_path=str(embedding_path) if embedding_path else None,
    )
    _save_profile(profile, avatar_dir)
    return profile


def load_avatar_profile(avatar_id: str) -> AvatarProfile:
    profile_path = AVATARS_DIR / avatar_id / "profile.json"
    if not profile_path.exists():
        raise AvatarNotReadyError(f"Аватар {avatar_id} не найден")
    data = json.loads(profile_path.read_text())
    return AvatarProfile(**data)


def _extract_reference_frames(source_path: Path, avatar_dir: Path) -> list[Path]:
    """
    Для видео-источника: вытаскивает несколько кадров с разных секунд, чтобы
    покрыть небольшой разброс ракурсов/мимики — это опорный набор,
    к которому липсинк-модель "привязывается" на каждом сегменте.
    Для фото-источника: единственный кадр используется как есть.
    """
    frames_dir = avatar_dir / "reference_frames"
    frames_dir.mkdir(exist_ok=True)

    if source_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
        dest = frames_dir / f"frame_0{source_path.suffix}"
        shutil.copy(source_path, dest)
        return [dest]

    if source_path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
        from .stitch import probe_duration, run_ffmpeg

        duration = probe_duration(str(source_path))
        # Берём до 5 кадров равномерно по длительности видео, не заходя
        # в самый конец (последняя секунда часто смазана/обрезана).
        n_frames = min(5, max(1, int(duration)))
        timestamps = [
            duration * (i + 1) / (n_frames + 1) for i in range(n_frames)
        ]

        frames = []
        for i, ts in enumerate(timestamps):
            out = frames_dir / f"frame_{i}.jpg"
            run_ffmpeg([
                "-ss", f"{ts:.2f}",
                "-i", str(source_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(out),
            ])
            frames.append(out)
        return frames

    raise ValueError(
        f"Неподдерживаемый формат файла: {source_path.suffix}. "
        f"Ожидается фото (jpg/png/webp) или видео (mp4/mov/avi/mkv)."
    )


def _build_face_embedding(reference_frames: list[Path], avatar_dir: Path) -> Path:
    """
    Строит усреднённый эмбеддинг лица по всем опорным кадрам через
    InsightFace (buffalo_l). Если на каком-то кадре не нашлось лица —
    кадр просто пропускается, а не валит весь процесс: для устойчивости
    важно иметь хотя бы один валидный эмбеддинг, а не идеальный набор.
    """
    import cv2
    import numpy as np

    analyzer = _get_face_analyzer()
    embeddings = []

    for frame_path in reference_frames:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        faces = analyzer.get(image)
        if not faces:
            continue
        # Если на кадре несколько лиц — берём самое крупное (обычно это
        # и есть основной субъект, а не случайный человек на фоне).
        largest_face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        embeddings.append(largest_face.embedding)

    if not embeddings:
        raise NoFaceDetectedError(
            "Не удалось распознать лицо ни на одном из кадров. "
            "Убедитесь, что лицо видно чётко, анфас, при хорошем освещении."
        )

    averaged = np.mean(embeddings, axis=0)
    embedding_path = avatar_dir / "embedding.npy"
    np.save(embedding_path, averaged)
    return embedding_path


def _save_profile(profile: AvatarProfile, avatar_dir: Path) -> None:
    (avatar_dir / "profile.json").write_text(
        json.dumps(profile.to_json(), ensure_ascii=False, indent=2)
    )
