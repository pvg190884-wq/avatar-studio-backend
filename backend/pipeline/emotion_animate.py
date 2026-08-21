"""
Модуль анимации фото-аватара эмоцией: перенос мимики + лёгкого движения
головы/плеч с видео-драйвера эмоции на статичное фото пользователя,
с сохранением идентичности лица и качества исходного изображения.

Пайплайн (почему именно так):

1. Детекция лица (InsightFace, тот же анализатор, что и в avatar.py —
   не грузим модель повторно, не плодим рассинхрон).
2. Выровненный кроп 512×512 вокруг головы+плеч по 5 ключевым точкам лица
   (а не грубый resize всего кадра, как в исходном demo.py TPSMM — именно
   грубый resize давал "волны"/размытие в первом тесте: лицо на исходном
   фото занимало малую часть кадра, и сжатие всего кадра до 256×256
   делало лицо микроскопическим и нечитаемым для модели).
3. TPSMM (Thin-Plate Spline Motion Model, ONNX) переносит движение с
   видео-драйвера эмоции на этот кроп. Модель работает фиксированно на
   256×256 — это архитектурное ограничение конкретных весов, обойти
   нельзя, поэтому кроп 512 перед этим уменьшается до 256 для подачи
   в модель.
4. GFPGAN восстанавливает резкость и одновременно поднимает разрешение
   обратно с 256 до 512 — это компенсирует потерю детализации на шаге 3
   и было тем самым недостающим звеном, из-за которого первый тест
   получился "размытым".
5. Итоговый анимированный кроп 512×512 вклеивается обратно в исходное
   фото ПОЛНОГО разрешения по обратной аффинной матрице — стол, ноутбук,
   вывеска на фоне остаются нетронутыми оригинальными пикселями, качество
   исходного фото вне зоны лица не теряется вообще.

Результат этого модуля — "молчащее" видео (голова+плечи оживлены
мимикой/лёгким движением, тело статично), которое дальше в pipeline
скармливается в lipsync.py как video_path для MuseTalk.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from scipy.spatial import ConvexHull
from tqdm import tqdm

from .avatar import _get_face_analyzer, NoFaceDetectedError

# backend/pipeline/emotion_animate.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
TPSMM_DIR = BACKEND_DIR / "TPSMM"
KP_DETECTOR_PATH = TPSMM_DIR / "checkpoints" / "kp_detector.onnx"
TPSMM_MODEL_PATH = TPSMM_DIR / "checkpoints" / "tpsmm_rel.onnx"

GFPGAN_DIR = BACKEND_DIR / "GFPGAN"
GFPGAN_MODEL_PATH = GFPGAN_DIR / "gfpgan-v1.4.onnx"

TPSMM_SIZE = 256  # фиксированный вход/выход модели, не меняется
CROP_SIZE = 512  # размер кропа лица для выравнивания/вклейки — совпадает
# с нативным разрешением GFPGAN, чтобы не терять резкость на лишних
# передискретизациях

# Референсные точки ArcFace-выравнивания (5 landmarks: глаза, нос, углы
# рта) для канвы 112×112 — стандарт, на котором обучались модели детекции
# лица вроде InsightFace. Расширяем/масштабируем эти точки под нужный
# размер кропа и добавляем отступ, чтобы попали не только глаза/нос/рот,
# но и лоб, подбородок, плечи.
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _imread_unicode(path: str) -> np.ndarray | None:
    """
    Замена cv2.imread для путей с не-ASCII символами (кириллица и т.п.).

    cv2.imread на Windows открывает файл через системный fopen в текущей
    кодовой странице ОС и молча возвращает None, если путь содержит
    символы вне этой кодировки — типичный случай: загруженные файлы вида
    "ChatGPT Image 2 июл. 2026 г., ....jpg". Читаем байты штатными
    средствами Python (которые Unicode-путь понимают всегда), а затем
    декодируем изображение из памяти — так путь вообще не участвует
    в работе OpenCV.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@dataclass
class AlignedCrop:
    image: np.ndarray  # вырезанный и выровненный кроп (BGR, CROP_SIZE x CROP_SIZE)
    matrix: np.ndarray  # аффинная матрица: оригинал -> кроп (нужна для обратной вклейки)


def _detect_face_kps(image: np.ndarray) -> np.ndarray:
    """
    Находит на изображении крупнейшее лицо и возвращает его 5 ключевых
    точек (kps) через уже существующий InsightFace-анализатор из
    avatar.py — не грузим вторую копию модели.
    """
    analyzer = _get_face_analyzer()
    faces = analyzer.get(image)
    if not faces:
        raise NoFaceDetectedError(
            "Не удалось найти лицо на фото для анимации эмоции. "
            "Убедитесь, что лицо видно чётко и анфас."
        )
    largest_face = max(
        faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    return largest_face.kps.astype(np.float32)


def get_aligned_crop(
    image: np.ndarray,
    size: int = CROP_SIZE,
    crop_scale: float = 2.2,
    vertical_shift: float = 0.18,
) -> AlignedCrop:
    """
    Вырезает и выравнивает область "голова + плечи" вокруг найденного
    лица. crop_scale > 1 расширяет зону вокруг лица (чем больше — тем
    шире область, тем больше видно плеч/фона вокруг). vertical_shift
    сдвигает кроп вниз, чтобы под подбородком осталось место для плеч,
    а не обрезалось прямо по линии рта, как в жёстком face-recognition
    кропе.

    Оба параметра эмпирические — их может понадобиться подстроить по
    результату первого теста (см. инструкцию по тестированию ниже).
    """
    kps = _detect_face_kps(image)

    ratio = size / 112.0
    dst = _ARCFACE_DST * ratio
    center = dst.mean(axis=0)
    dst = (dst - center) / crop_scale + center
    dst[:, 1] += size * vertical_shift

    # ВАЖНО: полный SimilarityTransform (поворот+масштаб+сдвиг, 4 степени
    # свободы) по 5 зашумлённым точкам landmarks оказался нестабилен
    # именно на этом типе фото (AI-сгенерированные изображения) — давал
    # паразитный поворот ~15° даже на визуально прямом анфас-лице,
    # что при обратной вклейке в полноразмерное фото превращалось в
    # заметный перекос "ромбом". Для аватар-фото (почти всегда анфас,
    # без сильного наклона головы) это ненужная степень свободы —
    # убираем поворот полностью и выравниваем только по масштабу и
    # сдвигу, используя положение глаз (самые стабильно детектируемые
    # из 5 точек). Это устраняет нестабильность в корне, а не лечит её
    # косвенно через штрафы/эвристики.
    left_eye, right_eye = kps[0], kps[1]
    eye_center = (left_eye + right_eye) / 2.0
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 1.0:
        raise NoFaceDetectedError(
            "Расстояние между глазами на фото аномально мало — "
            "детекция лица, вероятно, ошиблась."
        )

    dst_left_eye, dst_right_eye = dst[0], dst[1]
    dst_eye_center = (dst_left_eye + dst_right_eye) / 2.0
    dst_eye_dist = float(np.linalg.norm(dst_right_eye - dst_left_eye))

    scale = dst_eye_dist / eye_dist
    tx = dst_eye_center[0] - scale * eye_center[0]
    ty = dst_eye_center[1] - scale * eye_center[1]
    matrix = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
    cropped = cv2.warpAffine(image, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)
    return AlignedCrop(image=cropped, matrix=matrix)


def paste_back(
    original: np.ndarray,
    processed_crop: np.ndarray,
    matrix: np.ndarray,
    erode_px: int = 15,
) -> np.ndarray:
    """
    Вклеивает обработанный (анимированный+восстановленный) кроп обратно
    в оригинальное фото полного разрешения по обратной матрице.

    Используем cv2.seamlessClone (Poisson blending) вместо простого
    альфа-смешения с растушёвкой маски: обычное альфа-смешение убирает
    резкость границы, но не устраняет лёгкое тональное/яркостное
    несовпадение между вклеенным кропом и окружением (например, из-за
    двойной передискретизации при прямом+обратном warpAffine) — именно
    это несовпадение и оставалось видно как "квадрат" даже при сильной
    растушёвке. seamlessClone выравнивает и границу, и цвет/освещение
    одновременно — стандартный инструмент именно для такой вклейки.

    Работает только для не повёрнутых (осесимметричных) вставок — у нас
    это гарантировано, т.к. get_aligned_crop() намеренно не использует
    поворот (см. комментарий там).
    """
    h, w = original.shape[:2]
    inv_matrix = cv2.invertAffineTransform(matrix)

    warped_back = cv2.warpAffine(
        processed_crop, inv_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    mask = np.full(processed_crop.shape[:2], 255, dtype=np.uint8)
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        mask = cv2.erode(mask, kernel)

    mask_warped = cv2.warpAffine(
        mask, inv_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    mask_binary = (mask_warped > 127).astype(np.uint8) * 255

    ys, xs = np.where(mask_binary > 0)
    if len(xs) == 0:
        # Вырожденный случай (маска пустая) — просто вернуть оригинал без изменений,
        # чем падать с ошибкой.
        return original.copy()
    center = (int(xs.mean()), int(ys.mean()))

    try:
        return cv2.seamlessClone(warped_back, original, mask_binary, center, cv2.NORMAL_CLONE)
    except cv2.error:
        # seamlessClone иногда падает на вырожденных/слишком маленьких масках —
        # в этом случае откатываемся на простое альфа-смешение с растушёвкой,
        # чтобы пайплайн не падал целиком из-за одного кадра.
        mask_blurred = cv2.GaussianBlur(mask_binary, (0, 0), sigmaX=40)
        mask_f = (mask_blurred.astype(np.float32) / 255.0)[..., None]
        blended = warped_back.astype(np.float32) * mask_f + original.astype(np.float32) * (1 - mask_f)
        return blended.astype(np.uint8)


class TPSMMAnimator:
    """
    Обёртка над ONNX-сессиями TPSMM. Держит модели загруженными между
    кадрами одного видео (а не пересоздаёт сессию на каждый кадр) —
    инициализация ONNX Runtime не бесплатна, особенно на CPU.
    """

    def __init__(self):
        if not KP_DETECTOR_PATH.exists() or not TPSMM_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Не найдены веса TPSMM. Ожидались файлы:\n"
                f"  {KP_DETECTOR_PATH}\n  {TPSMM_MODEL_PATH}\n"
                f"Скачай их из https://github.com/instant-high/"
                f"Thin-plate-spline-motion-model-ONNX/releases"
            )
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        providers = ["CPUExecutionProvider"]
        self.kp_detector = onnxruntime.InferenceSession(
            str(KP_DETECTOR_PATH), sess_options=session_options, providers=providers
        )
        self.tpsm_model = onnxruntime.InferenceSession(
            str(TPSMM_MODEL_PATH), sess_options=session_options, providers=providers
        )

    def _get_kp(self, frame_rgb_float_chw: np.ndarray) -> np.ndarray:
        ort_inputs = {self.kp_detector.get_inputs()[0].name: frame_rgb_float_chw}
        return self.kp_detector.run(
            [self.kp_detector.get_outputs()[0].name], ort_inputs
        )[0]

    @staticmethod
    def _relative_kp(kp_source, kp_driving, kp_driving_initial, amplitude_scale: float = 1.0):
        """
        amplitude_scale < 1.0 приглушает перенесённое движение — вместо
        полной амплитуды эмоции с драйвера (нейтраль -> широкая улыбка)
        переносится только её часть. Нужно на CPU-версии TPSMM: при
        полной амплитуде на резких переходах мимики модель "заламывает"
        ткань изображения (артефакт thin-plate spline warping при
        деформации, выходящей за пределы того, что модель способна
        аккуратно интерполировать) — визуально выглядит как раздвоение/
        неестественный нарост в районе рта. Приглушённая амплитуда даёт
        менее выразительную, но структурно чистую мимику.
        """
        source_area = ConvexHull(kp_source[0]).volume
        driving_area = ConvexHull(kp_driving_initial[0]).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
        kp_value_diff = (kp_driving - kp_driving_initial) * adapt_movement_scale * amplitude_scale
        return kp_value_diff + kp_source

    @staticmethod
    def _center_crop_square(frame_bgr: np.ndarray) -> np.ndarray:
        """
        Обрезает кадр по центру до квадрата (без искажения пропорций),
        прежде чем ужимать до TPSMM_SIZE. Наши облегчённые видео-драйверы
        эмоций имеют соотношение сторон 16:9 (после ffmpeg-оптимизации),
        а TPSMM ждёт строго квадратный вход — прямой resize без кропа
        сжимал бы кадр по горизонтали, искажая пропорции лица на
        драйвере. Модель считывала это искажение как настоящее движение,
        что и давало эффект "голова норовит оторваться" — не баг модели,
        а искажённые входные данные.
        """
        h, w = frame_bgr.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return frame_bgr[y0:y0 + side, x0:x0 + side]

    @staticmethod
    def _prep(frame_bgr: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 -> RGB CHW float32 [0,1], батч-размерность добавлена."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (TPSMM_SIZE, TPSMM_SIZE)).astype("float32") / 255.0
        return np.transpose(rgb[np.newaxis], (0, 3, 1, 2))

    def animate(self, source_crop_bgr: np.ndarray, driving_frames_bgr: list[np.ndarray], amplitude_scale: float = 1.0) -> list[np.ndarray]:
        """
        source_crop_bgr: выровненный кроп лица (CROP_SIZE x CROP_SIZE, BGR)
        driving_frames_bgr: кадры видео-драйвера эмоции (любой размер, BGR)

        Возвращает список кадров TPSMM_SIZE x TPSMM_SIZE (BGR uint8) —
        источник с перенесённым движением, ещё БЕЗ восстановления резкости
        (это следующий шаг, GFPGAN).
        """
        source_input = self._prep(source_crop_bgr)
        kp_source = self._get_kp(source_input)

        driving_frames_bgr = [self._center_crop_square(f) for f in driving_frames_bgr]
        kp_driving_initial = self._get_kp(self._prep(driving_frames_bgr[0]))

        results = []
        for frame in tqdm(driving_frames_bgr, desc="TPSMM"):
            driving_input = self._prep(frame)
            kp_driving = self._get_kp(driving_input)
            kp_norm = self._relative_kp(kp_source, kp_driving, kp_driving_initial, amplitude_scale)

            ort_inputs = {
                self.tpsm_model.get_inputs()[0].name: kp_source,
                self.tpsm_model.get_inputs()[1].name: source_input,
                self.tpsm_model.get_inputs()[2].name: kp_norm,
                self.tpsm_model.get_inputs()[3].name: driving_input,
            }
            out = self.tpsm_model.run(
                [self.tpsm_model.get_outputs()[0].name], ort_inputs
            )[0]

            frame_rgb = np.transpose(out.squeeze(), (1, 2, 0))
            frame_rgb = np.clip(frame_rgb * 255, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            results.append(frame_bgr)
        return results


class GFPGANRestorer:
    """
    Обёртка над ONNX-сессией GFPGAN — восстанавливает резкость и поднимает
    разрешение 256 -> 512. Нормализация входа/выхода в диапазон [-1, 1] —
    стандарт для этого семейства моделей (StyleGAN2-based). Если на
    выходе получится подозрительно тёмная/пересвеченная картинка — это
    первый признак, что конкретно скачанный .onnx экспортирован с другой
    нормализацией, тогда нужно будет подстроить _preprocess/_postprocess
    по факту визуальной проверки.
    """

    INPUT_SIZE = 512

    def __init__(self):
        if not GFPGAN_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Не найдены веса GFPGAN: {GFPGAN_MODEL_PATH}\n"
                f"Скачай gfpgan-v1.4.onnx из https://github.com/clibdev/"
                f"GFPGAN-onnxruntime-demo/releases"
            )
        self.session = onnxruntime.InferenceSession(
            str(GFPGAN_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        normalized = (rgb.astype("float32") / 127.5) - 1.0  # [-1, 1]
        return np.transpose(normalized[np.newaxis], (0, 3, 1, 2))

    def _postprocess(self, output: np.ndarray) -> np.ndarray:
        restored = np.transpose(output.squeeze(), (1, 2, 0))
        restored = np.clip((restored + 1.0) / 2.0, 0, 1) * 255.0
        rgb = restored.astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def restore(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Восстанавливает один кадр (любого входного размера) до 512x512."""
        ort_inputs = {self.input_name: self._preprocess(frame_bgr)}
        output = self.session.run([self.output_name], ort_inputs)[0]
        return self._postprocess(output)


def _read_video_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _write_video(frames: list[np.ndarray], output_path: str, fps: float = 15.0) -> None:
    if not frames:
        raise ValueError("Нет кадров для записи видео")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)
    for frame in frames:
        writer.write(frame)
    writer.release()


def animate_avatar_with_emotion(
    photo_path: str,
    driver_video_path: str,
    output_path: str,
    crop_scale: float = 2.2,
    vertical_shift: float = 0.18,
    amplitude_scale: float = 1.0,
) -> str:
    """
    Главная функция пайплайна: фото + видео-драйвер эмоции -> видео с
    оживлённым лицом/плечами на фоне оригинального неизменного фото.

    Возвращает путь к итоговому видео (тот же, что output_path).
    """
    photo = _imread_unicode(photo_path)
    if photo is None:
        raise ValueError(f"Не удалось прочитать фото: {photo_path}")

    driving_frames = _read_video_frames(driver_video_path)
    if not driving_frames:
        raise ValueError(f"Не удалось прочитать кадры видео-драйвера: {driver_video_path}")

    aligned = get_aligned_crop(photo, size=CROP_SIZE, crop_scale=crop_scale, vertical_shift=vertical_shift)

    animator = TPSMMAnimator()
    animated_crops_256 = animator.animate(aligned.image, driving_frames, amplitude_scale=amplitude_scale)

    restorer = GFPGANRestorer()
    final_frames = []
    for crop_256 in tqdm(animated_crops_256, desc="GFPGAN restore"):
        restored_512 = restorer.restore(crop_256)
        pasted = paste_back(photo, restored_512, aligned.matrix)
        final_frames.append(pasted)

    fps = cv2.VideoCapture(driver_video_path).get(cv2.CAP_PROP_FPS) or 15.0
    _write_video(final_frames, output_path, fps=fps)
    return output_path


def debug_crop_check(
    photo_path: str,
    output_prefix: str = "debug",
    crop_scale: float = 2.2,
    vertical_shift: float = 0.18,
) -> None:
    """
    Быстрая проверка геометрии выравнивания (секунды, не минуты) — без
    TPSMM и GFPGAN. Сохраняет три файла:
      {prefix}_kps.png     — оригинал с нарисованными точками лица
      {prefix}_crop.png    — сам выровненный кроп
      {prefix}_roundtrip.png — кроп, вклеенный обратно БЕЗ изменений
                                (если тут всё чисто и на своём месте —
                                 геометрия верна, проблему искать дальше
                                 по пайплайну; если уже тут перекос —
                                 проблема именно в выравнивании).
    """
    photo = _imread_unicode(photo_path)
    if photo is None:
        raise ValueError(f"Не удалось прочитать фото: {photo_path}")

    kps = _detect_face_kps(photo)
    kps_preview = photo.copy()
    for x, y in kps:
        cv2.circle(kps_preview, (int(x), int(y)), 6, (0, 0, 255), -1)
    cv2.imwrite(f"{output_prefix}_kps.png", kps_preview)

    aligned = get_aligned_crop(photo, size=CROP_SIZE, crop_scale=crop_scale, vertical_shift=vertical_shift)
    cv2.imwrite(f"{output_prefix}_crop.png", aligned.image)

    print("Матрица (original -> crop):")
    print(aligned.matrix)
    inv = cv2.invertAffineTransform(aligned.matrix)
    print("Обратная матрица (crop -> original):")
    print(inv)
    print(f"Размер оригинала (H,W): {photo.shape[:2]}")
    print(f"Размер кропа (H,W): {aligned.image.shape[:2]}")

    roundtrip = paste_back(photo, aligned.image, aligned.matrix)
    cv2.imwrite(f"{output_prefix}_roundtrip.png", roundtrip)

    print(f"Сохранено: {output_prefix}_kps.png, {output_prefix}_crop.png, {output_prefix}_roundtrip.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Тест анимации фото-аватара эмоцией")
    parser.add_argument("--photo", required=True, help="Путь к фото аватара")
    parser.add_argument("--driver", help="Путь к видео-драйверу эмоции (не нужен для --crop-only)")
    parser.add_argument("--output", default="animated_test.mp4", help="Путь для результата")
    parser.add_argument("--crop-scale", type=float, default=2.2)
    parser.add_argument("--vertical-shift", type=float, default=0.18)
    parser.add_argument("--amplitude-scale", type=float, default=1.0, help="0.0-1.0, доля переносимой амплитуды эмоции")
    parser.add_argument(
        "--crop-only", action="store_true",
        help="Только проверить геометрию выравнивания (секунды), без TPSMM/GFPGAN",
    )
    args = parser.parse_args()

    if args.crop_only:
        debug_crop_check(
            args.photo, crop_scale=args.crop_scale, vertical_shift=args.vertical_shift,
        )
    else:
        if not args.driver:
            raise SystemExit("--driver обязателен, если не указан --crop-only")
        result = animate_avatar_with_emotion(
            args.photo, args.driver, args.output,
            crop_scale=args.crop_scale, vertical_shift=args.vertical_shift,
            amplitude_scale=args.amplitude_scale,
        )
        print(f"Готово: {result}")
