import ast
import re
from pathlib import Path
from typing import Any

HEADER_RE = r"^(#{1,6})\s+.*$"


def split_oversized_interval(
    start: int, end: int, max_chunk_size: int
) -> list[tuple[int, int]]:
    intervals = []
    while start < end:
        piece_end = min(start + max_chunk_size, end)
        intervals.append((start, piece_end))
        start = piece_end
    return intervals


def find_boundaries(content: str) -> list[int]:
    boundaries = [0]
    for m in re.finditer(HEADER_RE, content, re.MULTILINE):
        level = len(m.group(1))
        if level <= 2 and boundaries[-1] != m.start():
            boundaries.append(m.start())
    if boundaries[-1] != len(content):
        boundaries.append(len(content))
    return boundaries


def extract_md_chunks(
    content: str, boundaries: list[int], max_chunk_size: int = 2000
) -> list[tuple[int, int, str]]:
    chunks = []
    for i in range(len(boundaries) - 1):
        for start, end in split_oversized_interval(
            boundaries[i], boundaries[i + 1], max_chunk_size
        ):
            chunks.append((start, end, content[start:end]))
    return chunks


def chunk_markdown_files(
    corpus_root: str | Path = "data/raw", max_chunk_size: int = 2000
) -> list[dict[str, Any]]:
    md_files = Path(corpus_root).rglob("*.md")
    all_chunks = []
    for file in md_files:
        try:
            content = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"Skipping {file}: {e}")
            continue
        boundaries = find_boundaries(content)
        for start, end, text in extract_md_chunks(
            content, boundaries, max_chunk_size
        ):
            all_chunks.append(
                {
                    "file_path": str(file),
                    "text": text,
                    "first_char_idx": start,
                    "last_char_idx": end,
                }
            )
    return all_chunks


def find_python_units(
    content: str,
) -> list[tuple[int, int, str | None, str, str | None]]:
    tree = ast.parse(content)
    line_starts = [0]
    for line in content.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    units: list[tuple[int, int, str | None, str, str | None]] = []
    leftover_start = None

    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if leftover_start is not None:
                units.append(
                    (
                        leftover_start,
                        line_starts[node.lineno - 1],
                        None,
                        "ModuleCode",
                        None,
                    )
                )
                leftover_start = None
            start_lineno = (
                node.decorator_list[0].lineno
                if node.decorator_list
                else node.lineno
            )
            start = line_starts[start_lineno - 1]
            assert node.end_lineno is not None
            end = line_starts[node.end_lineno]
            units.append(
                (
                    start,
                    end,
                    node.name,
                    type(node).__name__,
                    ast.get_docstring(node),
                )
            )
        elif leftover_start is None:
            leftover_start = line_starts[node.lineno - 1]

    if leftover_start is not None:
        units.append((leftover_start, len(content), None, "ModuleCode", None))

    return units


def extract_python_chunks(
    content: str,
    file_path: str,
    units: list[tuple[int, int, str | None, str, str | None]],
    max_chunk_size: int = 2000,
) -> list[dict[str, Any]]:
    chunks = []
    for start, end, name, kind, docstring in units:
        for piece_start, piece_end in split_oversized_interval(
            start, end, max_chunk_size
        ):
            chunks.append(
                {
                    "file_path": file_path,
                    "name": name,
                    "type": kind,
                    "code": content[piece_start:piece_end],
                    "docstring": docstring,
                    "first_char_idx": piece_start,
                    "last_char_idx": piece_end,
                }
            )
    return chunks


def chunk_python_file(
    content: str, file_path: str, max_chunk_size: int = 2000
) -> list[dict[str, Any]]:
    try:
        units = find_python_units(content)
    except (SyntaxError, ValueError):
        print(f"Error: Failed to parse the file path: {file_path}")
        return []
    return extract_python_chunks(content, file_path, units, max_chunk_size)


def chunk_python_files(
    corpus_root: str | Path = "data/raw", max_chunk_size: int = 2000
) -> list[dict[str, Any]]:
    py_files = Path(corpus_root).rglob("*.py")
    all_chunks = []
    for file in py_files:
        try:
            content = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"Skipping {file}: {e}")
            continue
        all_chunks.extend(
            chunk_python_file(content, str(file), max_chunk_size)
        )

    return all_chunks
