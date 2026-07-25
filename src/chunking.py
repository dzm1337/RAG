import re
from pathlib import Path

PY_FILES = Path("data/raw/vllm-0.10.1").rglob("*.py")
HEADER_RE = r"^(#{1,6})\s+.*$"


def find_boundaries(content: str) -> list[int]:
    boundaries = [0]

    for m in re.finditer(HEADER_RE, content, re.MULTILINE):
        level = len(m.group(1))
        if level <= 2:
            boundaries.append(m.start())

    boundaries.append(len(content))
    return boundaries


def extract_md_chunks(
    content: str,
    boundaries: list[int],
):
    chunks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        chunks.append(content[start:end])
    return chunks


def chunk_markdown_files() -> list[dict]:
    md_files = Path("data/raw/vllm-0.10.1").rglob("*.py")
    all_chunks = []
    for file in md_files:
        content = files.read_text(encoding="utf-8")
        boundaries = find_boundaries(content)
        chunks = extract_md_chunks(content, boundaries)

        for i in range(len(chunks)):
            all_chunks.append(
                {
                    "file_path": str(file),
                    "text": chunk[i],
                    "first_char_idx": boundaries[i],
                    "last_char_idx": boundaries[i + 1],
                }
            )
    return all_chunks
