from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic RL manifest from paired noisy/clean audio.")
    parser.add_argument("--noisy_dir", type=Path, required=True, help="Directory containing noisy audio files.")
    parser.add_argument("--clean_dir", type=Path, required=True, help="Directory containing clean audio files.")
    parser.add_argument("--output_manifest", type=Path, required=True, help="Output JSONL manifest path.")
    parser.add_argument(
        "--data_root", type=Path, default=None,
        help="Store audio paths relative to this root; omit to retain absolute paths. "
             "Use the same directory as SPEECH_DATA_ROOT during training.",
    )
    parser.add_argument("--split", type=str, required=True, help="Split name to write into the manifest.")
    parser.add_argument("--chunk_seconds", type=float, default=10.0, help="Fixed chunk size in seconds.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target waveform sample rate for chunking.")
    parser.add_argument(
        "--duration_tolerance_ms",
        type=float,
        default=50.0,
        help="Maximum tolerated noisy/clean duration mismatch in milliseconds.",
    )
    return parser.parse_args()


def discover_audio_files(root: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    files_by_stem: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        stem = path.stem
        resolved = path.resolve()
        if stem in files_by_stem:
            if stem not in duplicates:
                duplicates[stem] = [str(files_by_stem[stem])]
            duplicates[stem].append(str(resolved))
            continue
        files_by_stem[stem] = resolved

    return files_by_stem, duplicates


def format_duplicate_error(name: str, duplicates: dict[str, list[str]]) -> str:
    preview = []
    for stem in sorted(duplicates)[:10]:
        preview.append(f"{stem}: {duplicates[stem]}")
    suffix = "" if len(duplicates) <= 10 else f" ... and {len(duplicates) - 10} more"
    return f"Found duplicate stems in {name}: " + "; ".join(preview) + suffix


def get_resampled_num_samples(path: Path, sample_rate: int) -> int:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise ValueError(f"Invalid sample rate for {path}: {info.samplerate}")
    return int(round(info.frames * sample_rate / info.samplerate))


def chunk_boundaries(total_samples: int, chunk_samples: int) -> list[tuple[int, int]]:
    if total_samples <= 0:
        return []
    if total_samples <= chunk_samples:
        return [(0, total_samples)]

    starts = list(range(0, total_samples - chunk_samples + 1, chunk_samples))
    tail_start = total_samples - chunk_samples
    if not starts or starts[-1] != tail_start:
        starts.append(tail_start)

    return [(start, min(start + chunk_samples, total_samples)) for start in starts]


def samples_to_seconds(num_samples: int, sample_rate: int) -> float:
    return num_samples / float(sample_rate)


def samples_to_milliseconds(num_samples: int, sample_rate: int) -> int:
    return int(round(num_samples * 1000.0 / sample_rate))


def build_manifest(args: argparse.Namespace) -> tuple[list[dict], dict]:
    noisy_dir = args.noisy_dir.resolve()
    clean_dir = args.clean_dir.resolve()
    output_manifest = args.output_manifest.resolve()
    data_root = getattr(args, "data_root", None)
    data_root = None if data_root is None else data_root.expanduser().resolve()

    def audio_path(path):
        if data_root is None:
            return str(path)
        try:
            return path.relative_to(data_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Audio file {path} is outside --data_root {data_root}") from exc

    noisy_files, noisy_duplicates = discover_audio_files(noisy_dir)
    clean_files, clean_duplicates = discover_audio_files(clean_dir)

    if noisy_duplicates:
        raise ValueError(format_duplicate_error("noisy_dir", noisy_duplicates))
    if clean_duplicates:
        raise ValueError(format_duplicate_error("clean_dir", clean_duplicates))

    noisy_stems = set(noisy_files)
    clean_stems = set(clean_files)
    common_stems = sorted(noisy_stems & clean_stems)
    missing_in_noisy = sorted(clean_stems - noisy_stems)
    missing_in_clean = sorted(noisy_stems - clean_stems)

    tolerance_samples = int(round(args.duration_tolerance_ms * args.sample_rate / 1000.0))
    chunk_samples = int(round(args.chunk_seconds * args.sample_rate))
    if chunk_samples <= 0:
        raise ValueError(f"`chunk_seconds` must be positive, got {args.chunk_seconds}")

    records: list[dict] = []
    duration_mismatches: list[dict] = []
    short_source_count = 0
    multi_chunk_source_count = 0

    for stem in common_stems:
        noisy_path = noisy_files[stem]
        clean_path = clean_files[stem]
        noisy_samples = get_resampled_num_samples(noisy_path, args.sample_rate)
        clean_samples = get_resampled_num_samples(clean_path, args.sample_rate)

        if abs(noisy_samples - clean_samples) > tolerance_samples:
            duration_mismatches.append(
                {
                    "stem": stem,
                    "noisy_path": audio_path(noisy_path),
                    "clean_path": audio_path(clean_path),
                    "noisy_num_samples": noisy_samples,
                    "clean_num_samples": clean_samples,
                }
            )
            continue

        usable_samples = min(noisy_samples, clean_samples)
        boundaries = chunk_boundaries(usable_samples, chunk_samples)
        if not boundaries:
            continue

        if usable_samples <= chunk_samples:
            short_source_count += 1
        if len(boundaries) > 1:
            multi_chunk_source_count += 1

        for chunk_index, (start_sample, end_sample) in enumerate(boundaries):
            start_ms = samples_to_milliseconds(start_sample, args.sample_rate)
            end_ms = samples_to_milliseconds(end_sample, args.sample_rate)
            record = {
                "utt_id": f"{stem}_{start_ms:09d}_{end_ms:09d}",
                "source_utt_id": stem,
                "noisy_path": audio_path(noisy_path),
                "clean_path": audio_path(clean_path),
                "text": "",
                "split": args.split,
                "chunk_index": chunk_index,
                "chunk_start_sec": samples_to_seconds(start_sample, args.sample_rate),
                "chunk_end_sec": samples_to_seconds(end_sample, args.sample_rate),
                "duration_sec": samples_to_seconds(end_sample - start_sample, args.sample_rate),
            }
            records.append(record)

    records.sort(key=lambda item: (item["source_utt_id"], item["chunk_index"], item["utt_id"]))

    stats = {
        "audio_path_base": "absolute" if data_root is None else "SPEECH_DATA_ROOT",
        "data_root": None if data_root is None else str(data_root),
        "noisy_dir": audio_path(noisy_dir),
        "clean_dir": audio_path(clean_dir),
        "output_manifest": str(output_manifest),
        "split": args.split,
        "sample_rate": args.sample_rate,
        "chunk_seconds": args.chunk_seconds,
        "duration_tolerance_ms": args.duration_tolerance_ms,
        "num_noisy_files": len(noisy_files),
        "num_clean_files": len(clean_files),
        "num_common_stems": len(common_stems),
        "num_missing_in_noisy": len(missing_in_noisy),
        "num_missing_in_clean": len(missing_in_clean),
        "num_duration_mismatches": len(duration_mismatches),
        "num_short_sources": short_source_count,
        "num_multi_chunk_sources": multi_chunk_source_count,
        "num_source_utts": len({record["source_utt_id"] for record in records}),
        "num_manifest_rows": len(records),
        "missing_in_noisy_examples": missing_in_noisy[:100],
        "missing_in_clean_examples": missing_in_clean[:100],
        "duration_mismatch_examples": duration_mismatches[:100],
    }
    return records, stats


def write_outputs(records: list[dict], stats: dict, output_manifest: Path) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    stats_path = output_manifest.with_name(f"{output_manifest.stem}.stats.json")

    with output_manifest.open("w", encoding="utf-8") as manifest_file:
        for record in records:
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with stats_path.open("w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    records, stats = build_manifest(args)
    write_outputs(records, stats, args.output_manifest.resolve())
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
