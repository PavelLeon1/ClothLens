"""Export a compact project snapshot to a Markdown file."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_OUTPUT = "project_context.md"
DEFAULT_MAX_FILE_SIZE_KB = 512

EXCLUDED_DIRS = {
    ".agents",
    ".codex",
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "checkpoints",
    "data",
    "dist",
    "htmlcov",
    "models",
    "node_modules",
    "results",
    "venv",
    "wandb",
}

EXCLUDED_FILE_NAMES = {
    ".coverage",
    "AGENT.md",
    "AGENTS.md",
    "Thumbs.db",
}

EXCLUDED_SUFFIXES = {
    ".7z",
    ".bin",
    ".bmp",
    ".ckpt",
    ".db",
    ".gif",
    ".jpeg",
    ".jpg",
    ".onnx",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".sqlite",
    ".sqlite3",
    ".webp",
    ".zip",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".env",
    ".gitignore",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

LANGUAGE_BY_SUFFIX = {
    ".css": "css",
    ".html": "html",
    ".js": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the project tree and source files into one Markdown file "
            "for report-writing context."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root. Defaults to the repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Markdown file to create. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--max-file-size-kb",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE_KB,
        help=(
            "Skip text files larger than this size. "
            f"Defaults to {DEFAULT_MAX_FILE_SIZE_KB} KB."
        ),
    )
    return parser.parse_args()


def should_skip_dir(path: Path) -> bool:
    return path.name in EXCLUDED_DIRS


def should_skip_file(path: Path, output_path: Path, max_file_size: int) -> bool:
    if path.resolve() == output_path.resolve():
        return True
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    if path.name.startswith("project_context") and path.suffix == ".md":
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if path.stat().st_size > max_file_size:
        return True
    return not is_text_file(path)


def is_text_file(path: Path) -> bool:
    if path.name in {".gitignore", ".env.example"}:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def iter_source_files(
    root: Path,
    output_path: Path,
    max_file_size: int,
) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if any(parent.name in EXCLUDED_DIRS for parent in path.parents):
            continue
        if path.is_dir():
            continue
        if should_skip_file(path, output_path, max_file_size):
            continue
        files.append(path)
    return files


def visible_children(
    directory: Path,
    output_path: Path,
    max_file_size: int,
) -> list[Path]:
    children: list[Path] = []
    for child in sorted(
        directory.iterdir(),
        key=lambda item: (not item.is_dir(), item.name),
    ):
        if child.is_dir():
            if should_skip_dir(child):
                continue
            if has_visible_descendants(child, output_path, max_file_size):
                children.append(child)
        elif not should_skip_file(child, output_path, max_file_size):
            children.append(child)
    return children


def has_visible_descendants(
    directory: Path,
    output_path: Path,
    max_file_size: int,
) -> bool:
    for child in directory.rglob("*"):
        if any(parent.name in EXCLUDED_DIRS for parent in child.parents):
            continue
        if child.is_file() and not should_skip_file(child, output_path, max_file_size):
            return True
    return False


def build_tree(root: Path, output_path: Path, max_file_size: int) -> str:
    lines = [f"{root.name}/"]

    def add_directory(directory: Path, prefix: str) -> None:
        children = visible_children(directory, output_path, max_file_size)
        for index, child in enumerate(children):
            is_last = index == len(children) - 1
            connector = "`-- " if is_last else "|-- "
            lines.append(f"{prefix}{connector}{child.name}")
            if child.is_dir():
                next_prefix = f"{prefix}{'    ' if is_last else '|   '}"
                add_directory(child, next_prefix)

    add_directory(root, "")
    return "\n".join(lines)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def fence_language(path: Path) -> str:
    if path.name == ".gitignore":
        return "gitignore"
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")


def render_markdown(root: Path, output_path: Path, max_file_size: int) -> str:
    files = iter_source_files(root, output_path, max_file_size)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# ClothLens project context",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "This file contains the project tree and source text files for report "
        "writing or review by another AI assistant.",
        "",
        "Skipped by default: virtual environments, Git internals, generated "
        "data/results, model checkpoints, images, binaries, caches, and local "
        "AI context files.",
        "",
        "## Project tree",
        "",
        "```text",
        build_tree(root, output_path, max_file_size),
        "```",
        "",
        "## Files",
        "",
    ]

    for path in files:
        relative_path = path.relative_to(root).as_posix()
        lines.extend(
            [
                f"### `{relative_path}`",
                "",
                f"```{fence_language(path)}",
                read_text(path).rstrip(),
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    max_file_size = args.max_file_size_kb * 1024

    markdown = render_markdown(root, output_path, max_file_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Project context exported to {output_path}")


if __name__ == "__main__":
    main()
