"""
Модуль сборки нескольких коротких сегментов в один "фильм".

Это единственный модуль пайплайна, который не требует GPU и работает
полностью уже сейчас — он собран на ffmpeg. Между сегментами видео и
аудио плавные кроссфейды (xfade/acrossfade), а не жёсткая склейка.
Это то, что отличает "фильм" от "видео, приклеенных друг к другу".
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

DEFAULT_TRANSITION_SEC = 0.6


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def probe_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def stitch_segments(
    segment_paths: list[str],
    output_path: str,
    transition_sec: float = DEFAULT_TRANSITION_SEC,
) -> str:
    """
    Склеивает список видео-сегментов в один файл с кроссфейдом между ними.
    Для 2+ сегментов строит цепочку xfade/acrossfade фильтров.
    Для 1 сегмента — просто копирует файл (без перекодирования).
    """
    if not segment_paths:
        raise ValueError("Нужен хотя бы один сегмент")

    if len(segment_paths) == 1:
        run_ffmpeg(["-i", segment_paths[0], "-c", "copy", output_path])
        return output_path

    durations = [probe_duration(p) for p in segment_paths]

    inputs = []
    for p in segment_paths:
        inputs += ["-i", p]

    # Строим цепочку xfade (видео) и acrossfade (аудио) последовательно:
    # [0][1]xfade=...[v01]; [v01][2]xfade=...[v012]; ...
    filter_parts = []
    running_offset = durations[0] - transition_sec
    prev_v = "0:v"
    prev_a = "0:a"

    for i in range(1, len(segment_paths)):
        vout = f"v{i}" if i < len(segment_paths) - 1 else "vout"
        aout = f"a{i}" if i < len(segment_paths) - 1 else "aout"

        filter_parts.append(
            f"[{prev_v}][{i}:v]xfade=transition=fade:"
            f"duration={transition_sec}:offset={running_offset:.3f}[{vout}]"
        )
        filter_parts.append(
            f"[{prev_a}][{i}:a]acrossfade=d={transition_sec}[{aout}]"
        )
        prev_v, prev_a = vout, aout
        if i < len(durations) - 1:
            running_offset += durations[i] - transition_sec

    filter_complex = ";".join(filter_parts)

    args = inputs + [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]
    run_ffmpeg(args)
    return output_path


def build_project_manifest(
    project_id: str,
    segment_paths: list[str],
    output_dir: str,
) -> dict:
    """Метаданные проекта-фильма: порядок сегментов, длительности, итог."""
    durations = [probe_duration(p) for p in segment_paths]
    total_raw = sum(durations)
    total_with_transitions = total_raw - DEFAULT_TRANSITION_SEC * (len(segment_paths) - 1)

    manifest = {
        "project_id": project_id,
        "segments": [
            {"path": p, "duration_sec": round(d, 2)}
            for p, d in zip(segment_paths, durations)
        ],
        "total_duration_sec": round(max(total_with_transitions, 0), 2),
        "output_dir": output_dir,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    return manifest


def concat_video_chunks(chunk_paths: list[str], output_path: str) -> str:
    """
    Склеивает чанки одного сегмента встык (без кроссфейда — это один
    непрерывный кусок речи, а не переход между разными сценами).
    Использует ffmpeg concat demuxer с перекодировкой, т.к. чанки из
    разных запусков MuseTalk не гарантированно идентичны по параметрам
    контейнера для потокового -c copy.
    """
    if len(chunk_paths) == 1:
        shutil.copy(chunk_paths[0], output_path)
        return output_path

    list_file = Path(output_path).with_suffix(".txt")
    list_file.write_text(
        "\n".join(f"file '{Path(p).resolve()}'" for p in chunk_paths),
        encoding="utf-8",
    )
    run_ffmpeg([
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ])
    list_file.unlink(missing_ok=True)
    return output_path