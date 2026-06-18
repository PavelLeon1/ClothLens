"""Create a portable balanced subset from a full DeepFashion2 dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
DEFAULT_CATEGORIES = ("top", "bottom", "dress", "outerwear")
DEEPFASHION2_CATEGORY_BY_ID = {
    1: "top",
    2: "top",
    3: "outerwear",
    4: "outerwear",
    5: "top",
    6: "top",
    7: "bottom",
    8: "bottom",
    9: "bottom",
    10: "dress",
    11: "dress",
    12: "dress",
    13: "dress",
}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    annotation_path: Path
    categories: tuple[str, ...]


@dataclass(frozen=True)
class SelectedSample:
    sample: Sample
    assigned_category: str


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/DeepFashion2")
    parser.add_argument("--output", default="data/raw/DeepFashion2_subset")
    parser.add_argument("--train-size", type=int, default=30_000)
    parser.add_argument("--validation-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        choices=list(DEFAULT_CATEGORIES),
    )
    parser.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--archive")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before writing the subset.",
    )
    return parser.parse_args(argv)


def has_polygon(segmentation: object) -> bool:
    if not isinstance(segmentation, list) or not segmentation:
        return False
    if all(isinstance(value, int | float) for value in segmentation):
        return len(segmentation) >= 6 and len(segmentation) % 2 == 0
    return any(
        isinstance(polygon, list)
        and len(polygon) >= 6
        and len(polygon) % 2 == 0
        for polygon in segmentation
    )


def read_target_categories(
    annotation_path: Path,
    allowed_categories: set[str],
) -> tuple[str, ...]:
    with annotation_path.open(encoding="utf-8") as annotation_file:
        annotation = json.load(annotation_file)

    categories: set[str] = set()
    if not isinstance(annotation, dict):
        return ()

    for key, item in annotation.items():
        if not key.startswith("item") or not isinstance(item, dict):
            continue
        category = DEEPFASHION2_CATEGORY_BY_ID.get(int(item.get("category_id", 0)))
        if category in allowed_categories and has_polygon(item.get("segmentation")):
            categories.add(category)
    return tuple(sorted(categories))


def collect_samples(
    source: Path,
    *,
    split: str,
    categories: tuple[str, ...],
) -> list[Sample]:
    image_dir = source / split / "image"
    annotation_dir = source / split / "annos"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing DeepFashion2 image directory: {image_dir}")
    if not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"Missing DeepFashion2 annotation directory: {annotation_dir}"
        )

    allowed_categories = set(categories)
    samples: list[Sample] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        annotation_path = annotation_dir / f"{image_path.stem}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Missing annotation for {image_path.name}: {annotation_path}"
            )
        sample_categories = read_target_categories(annotation_path, allowed_categories)
        if sample_categories:
            samples.append(
                Sample(
                    image_path=image_path,
                    annotation_path=annotation_path,
                    categories=sample_categories,
                )
            )
    return samples


def count_source_categories(samples: list[Sample]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for sample in samples:
        counter.update(sample.categories)
    return dict(sorted(counter.items()))


def select_balanced_samples(
    samples: list[Sample],
    *,
    requested_count: int,
    categories: tuple[str, ...],
    seed: int,
) -> list[SelectedSample]:
    if requested_count < 1:
        raise ValueError("Requested subset size must be greater than zero.")

    rng = random.Random(seed)
    buckets = {
        category: [sample for sample in samples if category in sample.categories]
        for category in categories
    }
    for bucket in buckets.values():
        rng.shuffle(bucket)

    positions = dict.fromkeys(categories, 0)
    selected_keys: set[str] = set()
    selected: list[SelectedSample] = []

    while len(selected) < requested_count:
        made_progress = False
        for category in categories:
            bucket = buckets[category]
            while positions[category] < len(bucket):
                sample = bucket[positions[category]]
                positions[category] += 1
                sample_key = sample.image_path.name
                if sample_key in selected_keys:
                    continue
                selected_keys.add(sample_key)
                selected.append(
                    SelectedSample(sample=sample, assigned_category=category)
                )
                made_progress = True
                break
            if len(selected) == requested_count:
                break
        if not made_progress:
            break

    if len(selected) != requested_count:
        raise ValueError(
            f"Requested {requested_count} samples, but only {len(selected)} unique "
            "samples are available for the selected categories."
        )
    return selected


def prepare_output(output: Path, *, clean: bool, source: Path) -> None:
    if output.resolve() == source.resolve():
        raise ValueError("Output directory must be different from source directory.")
    if output.exists() and clean:
        shutil.rmtree(output)
    for split in ("train", "validation"):
        (output / split / "image").mkdir(parents=True, exist_ok=True)
        (output / split / "annos").mkdir(parents=True, exist_ok=True)


def copy_or_link(source: Path, destination: Path, *, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def write_split(
    *,
    selected: list[SelectedSample],
    output: Path,
    split: str,
    mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selected_sample in selected:
        sample = selected_sample.sample
        image_destination = output / split / "image" / sample.image_path.name
        annotation_destination = (
            output / split / "annos" / sample.annotation_path.name
        )
        copy_or_link(sample.image_path, image_destination, mode=mode)
        copy_or_link(sample.annotation_path, annotation_destination, mode=mode)
        rows.append(
            {
                "image": f"{split}/image/{sample.image_path.name}",
                "annotation": f"{split}/annos/{sample.annotation_path.name}",
                "assigned_category": selected_sample.assigned_category,
                "categories": list(sample.categories),
            }
        )
    return rows


def split_manifest(
    *,
    selected: list[SelectedSample],
    source_samples: list[Sample],
    rows: list[dict[str, Any]],
    requested_count: int,
) -> dict[str, Any]:
    assigned_counter = Counter(
        selected_sample.assigned_category for selected_sample in selected
    )
    return {
        "requested_count": requested_count,
        "selected_count": len(selected),
        "available_count": len(source_samples),
        "source_category_counts": count_source_categories(source_samples),
        "assigned_category_counts": dict(sorted(assigned_counter.items())),
        "samples": rows,
    }


def write_archive(output: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zip_file:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zip_file.write(path, arcname=path.relative_to(output.parent))


def build_subset(
    *,
    source: Path,
    output: Path,
    train_size: int,
    validation_size: int,
    categories: tuple[str, ...],
    seed: int,
    mode: str,
    clean: bool,
    archive: Path | None,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    prepare_output(output, clean=clean, source=source)

    manifest: dict[str, Any] = {
        "source": str(source),
        "output": str(output),
        "seed": seed,
        "mode": mode,
        "categories": list(categories),
        "splits": {},
    }
    split_sizes = {"train": train_size, "validation": validation_size}
    for index, (split, requested_count) in enumerate(split_sizes.items()):
        source_samples = collect_samples(source, split=split, categories=categories)
        selected = select_balanced_samples(
            source_samples,
            requested_count=requested_count,
            categories=categories,
            seed=seed + index,
        )
        rows = write_split(
            selected=selected,
            output=output,
            split=split,
            mode=mode,
        )
        manifest["splits"][split] = split_manifest(
            selected=selected,
            source_samples=source_samples,
            rows=rows,
            requested_count=requested_count,
        )

    manifest_path = output / "subset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if archive is not None:
        write_archive(output, archive.resolve())
    return manifest


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    categories = tuple(arguments.categories)
    try:
        manifest = build_subset(
            source=Path(arguments.source),
            output=Path(arguments.output),
            train_size=arguments.train_size,
            validation_size=arguments.validation_size,
            categories=categories,
            seed=arguments.seed,
            mode=arguments.mode,
            clean=arguments.clean,
            archive=Path(arguments.archive) if arguments.archive else None,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Failed to prepare DeepFashion2 subset: {error}") from error

    for split, summary in manifest["splits"].items():
        print(
            f"{split}: selected {summary['selected_count']} "
            f"of {summary['available_count']} available samples"
        )
        print(f"{split}: {summary['assigned_category_counts']}")
    print(f"Subset written to: {manifest['output']}")
    if arguments.archive:
        print(f"Archive written to: {Path(arguments.archive).resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])
