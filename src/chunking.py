import ast
import re
from pathlib import Path

HEADER_RE = r"^(#{1,6})\s+.*$"


def find_boundaries(content: str) -> list[int]:
    boundaries = [0]

    for m in re.finditer(HEADER_RE, content, re.MULTILINE):
        level = len(m.group(1))
        if level <= 2 and boundaries[-1] != m.start():
            boundaries.append(m.start())
    if boundaries[-1] != len(content):
        boundaries.append(len(content))
    return boundaries


def split_oversized_span(
    start: int, end: int, max_chunk_size: int
) -> list[tuple[int, int]]:
    spans = []
    pos = start
    while pos < end:
        piece_end = min(pos + max_chunk_size, end)
        spans.append((pos, piece_end))
        pos = piece_end
    return spans


def extract_md_chunks(
    content: str,
    boundaries: list[int],
    max_chunk_size: int = 2000,
) -> list[tuple[int, int, str]]:
    chunks = []
    for i in range(len(boundaries) - 1):
        for start, end in split_oversized_span(
            boundaries[i], boundaries[i + 1], max_chunk_size
        ):
            chunks.append((start, end, content[start:end]))
    return chunks


def chunk_markdown_files(max_chunk_size: int = 2000) -> list[dict]:
    md_files = Path("data/raw/vllm-0.10.1").rglob("*.md")
    all_chunks = []
    for file in md_files:
        content = file.read_text(encoding="utf-8")
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


def chunk_python_file(
    content: str, file_path: str, max_chunk_size: int = 2000
) -> list[dict]:
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        print(f"Error: Failed to parse the file path: {file_path}")
        return []

    line_starts = [0]
    for line in content.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    chunks = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue

        docstring = ast.get_docstring(node)
        start_char = line_starts[node.lineno - 1]
        end_char = line_starts[node.end_lineno]

        for start, end in split_oversized_span(
            start_char, end_char, max_chunk_size
        ):
            chunks.append(
                {
                    "file_path": file_path,
                    "name": node.name,
                    "type": type(node).__name__,
                    "docstring": docstring,
                    "code": content[start:end],
                    "first_char_idx": start,
                    "last_char_idx": end,
                }
            )
    return chunks


def chunk_python_files(max_chunk_size: int = 2000) -> list[dict]:
    py_files = Path("data/raw/vllm-0.10.1").rglob("*.py")
    all_chunks = []
    for file in py_files:
        content = file.read_text(encoding="utf-8")
        chunks = chunk_python_file(content, str(file), max_chunk_size)
        all_chunks.extend(chunks)
    return all_chunks
