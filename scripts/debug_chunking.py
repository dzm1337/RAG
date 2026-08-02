import ast
from pathlib import Path
from pudb import set_trace

from src.chunking import (
    find_comment_start,
    to_corpus_path,
    build_chunk,
    chunk_corpus,
    chunk_markdown_content,
    chunk_markdown_files,
    chunk_python_content,
    chunk_python_source,
    chunk_python_files,
    parse_headers,
    build_line_starts,
    build_python_chunks,
    parse_python_units,
    build_header_path,
    iter_corpus_files,
    build_config,
    merge_small_sections,
    build_node_span,
    split_markdown_section,
    split_span,
)

MD_PATH = "data/raw/vllm-0.10.1/docs/features/lora.md"
PY_PATH = "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py"
CORPUS_ROOT = "data/raw"

MD_CONTENT = Path(MD_PATH).read_text(encoding="utf-8")
PY_CONTENT = Path(PY_PATH).read_text(encoding="utf-8")


def title(text: str) -> None:
    print("\n" + "#" * 72)
    print(f"# {text}")
    print("#" * 72)


def preview(value, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def demo_build_config() -> None:
    title("build_config -- turns raw CLI numbers into a validated ChunkConfig")
    for kwargs in [
        {},
        {"max_chunk_size": 500},
        {"max_chunk_size": 2000, "target_chunk_size": 1800},
        {"max_chunk_size": 2000, "target_chunk_size": 1800, "overlap": 50},
    ]:
        config = build_config(**kwargs)
        print(f"build_config({kwargs""})")
        print(f"  -> {config}")


def demo_split_span() -> None:

    title("split_span -- the only place that actually cuts text")
    print("A 25-char span, max_chunk_size=10, no overlap:")
    print(" ", split_span(0, 25, max_chunk_size=10))
    print("Same span, overlap=3 (each piece re-reads the tail of the last):")
    print(" ", split_span(0, 25, max_chunk_size=10, overlap=3))
    print("A span already smaller than max_chunk_size: returned untouched.")
    print(" ", split_span(0, 7, max_chunk_size=10))


def demo_iter_corpus_files() -> None:
    title(
        "iter_corpus_files -- walks corpus_root, filters by suffix"
        " + noise dirs"
    )
    from src.chunking import TEXT_SUFFIXES

    files = list(iter_corpus_files(CORPUS_ROOT, TEXT_SUFFIXES))
    print(f"{len(files)} text files found under {CORPUS_ROOT}. First 5:")
    for f in files[:10]:
        print(f"{f}")

def demo_to_corpus_path() -> None:
    title("to_corpus_path -- renders a path exactly as the grader expects it")
    p = Path(MD_PATH)
    print(f"to_corpus_path(Path({MD_PATH""})) -> {to_corpus_path(p)}")
    print("(relative to cwd if possible, else str(path) unchanged)")


def demo_parse_headers() -> None:
    title("parse_headers -- finds every ATX header, ignoring fenced code")
    headers = parse_headers(MD_CONTENT)
    print(f"{len(headers)} headers in {MD_PATH}:")
    for offset, level, text in headers:
        print(f"offset={offset}  h{level}  {text""}")


def demo_build_header_path() -> None:
    title(
        "build_header_path -- breadcrumb of headers at a given offset"
    )
    headers = parse_headers(MD_CONTENT)
    for pos in [0, 3000, 8000]:
        trail = build_header_path(headers, pos)
        print(f"build_header_path(headers, {pos}) -> {trail""}")


def demo_split_markdown_section() -> None:
    title(
        "split_markdown_section -- cuts on h1, then h2, then h3..."
    )
    headers = parse_headers(MD_CONTENT)
    config = build_config(max_chunk_size=2000)
    sections = split_markdown_section(
        headers, 0, len(MD_CONTENT), level=1, config=config
    )
    print(f"{len(sections)} sections (size={config.size}):")
    print(sections)
    for start, end in sections:
        print(f"[{start}:{end}]  len={end - start}")


def demo_merge_small_sections() -> None:
    title("merge_small_sections -- folds tiny sections into a neighbour")
    headers = parse_headers(MD_CONTENT)
    config = build_config(max_chunk_size=2000)
    sections = split_markdown_section(headers, 0, len(MD_CONTENT), 1, config)
    merged = merge_small_sections(sections, config)
    print(f"{len(sections)} sections before merge -> {len(merged)} after:")
    for start, end in merged:
        print(f"[{start}:{end}]  len={end - start}")


def demo_build_chunk() -> None:
    title(
        "build_chunk -- assembles one Chunk (text + enriched index_text)"
    )
    chunk = build_chunk(
        file_path=MD_PATH,
        content=MD_CONTENT,
        start=0,
        end=200,
        kind="Markdown",
        name="LoRA Adapters",
        context="LoRA Adapters",
    )
    for key, value in chunk.model_dump().items():
        print(f"{key}: {preview(value)}")


def demo_chunk_markdown_content() -> None:
    title(
        "chunk_markdown_content -- glues headers+sections+merge+build together"
    )
    config = build_config(max_chunk_size=2000)
    chunks = chunk_markdown_content(MD_CONTENT, MD_PATH, config)
    print(f"{len(chunks)} chunks produced for one file. First chunk:")
    for key, value in chunks[0].model_dump().items():
        print(f"{key}: {preview(value)}")


def demo_chunk_markdown_files() -> None:
    title(
        "chunk_markdown_files -- loops chunk_markdown_content over the corpus"
    )
    chunks = chunk_markdown_files(CORPUS_ROOT, max_chunk_size=2000)
    print(f"{len(chunks)} markdown/text chunks across the whole corpus.")
    print("Sample file_paths seen:")
    seen = {c.file_path for c in chunks}
    for fp in list(seen)[:5]:
        print(f"{fp}")


def demo_build_line_starts() -> None:
    title("build_line_starts -- char offset where each source line begins")
    sample = "import os\nimport sys\n\nprint('hi')\n"
    starts = build_line_starts(sample)
    print(f"source: {sample""}")
    print(f"line_starts: {starts}")
    print("  -> line 1 starts at 0, line 2 at", starts[1], ", etc.")


def demo_find_comment_start() -> None:
    title(
        "find_comment_start -- pulls a comment block into the def above it"
    )
    sample = (
        "x = 1\n# explains foo\n# still explaining foo\ndef foo():\n    pass\n"
    )
    line_starts = build_line_starts(sample)
    start = find_comment_start(sample, line_starts, lineno=4, floor=0)
    print(f"source:\n{sample}")
    print(f"find_comment_start(..., lineno=4, floor=0) -> offset {start}")
    print(f"span from there: {sample[start:]""}")


def demo_build_node_span() -> None:
    title(
        "build_node_span -- full span of a def: decorators + comments"
        " + body"
    )
    tree = ast.parse(PY_CONTENT)
    line_starts = build_line_starts(PY_CONTENT)
    decorated = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.decorator_list
    )
    start, end = build_node_span(decorated, PY_CONTENT, line_starts, floor=0)
    print(f"first decorated function found: {decorated.name""}")
    print(f"node.lineno (def line) = {decorated.lineno}")
    print(f"decorator lineno       = {decorated.decorator_list[0].lineno}")
    print(f"build_node_span -> [{start}:{end}], text starts with:")
    print(" ", preview(PY_CONTENT[start : start + 60]))


def demo_parse_units_and_parse_python_units() -> None:
    title("parse_units / parse_python_units -- partitions a file into Units")
    config = build_config(max_chunk_size=2000)
    units = parse_python_units(PY_CONTENT, config)
    print(f"{len(units)} units total. Kind breakdown:")
    kinds: dict[str, int] = {}
    for u in units:
        kinds[u.kind] = kinds.get(u.kind, 0) + 1
    for kind, count in sorted(kinds.items()):
        print(f"  {kind} {count}")

    print("\nFirst 5 units (start, end, name, kind):")
    for u in units[:5]:
        print(f"  [{u.start}:{u.end}]  {u.kind} name={u.name""}")

    print("\nA nested unit (Class.method) proves parse_units recurses:")
    nested = next((u for u in units if u.name and "." in u.name), None)
    if nested is None:
        print(f"  none: no class exceeds size={config.size}")
    else:
        print(f"  {nested.kind} {nested.name} [{nested.start}:{nested.end}]")


def demo_build_python_chunks() -> None:
    title(
        "build_python_chunks -- turns Units into chunk dicts (size-capped)"
    )
    config = build_config(max_chunk_size=2000)
    units = parse_python_units(PY_CONTENT, config)
    chunks = build_python_chunks(PY_CONTENT, PY_PATH, units, config)
    print(
        f"{len(units)} units -> {len(chunks)} chunks (oversized units split)"
    )
    named = next(c for c in chunks if c.name)
    print("One named chunk in full:")
    for key, value in named.model_dump().items():
        print(f"  {key}: {preview(value)}")


def demo_chunk_python_content_and_source() -> None:
    title("chunk_python_content / chunk_python_source -- one file, parse-safe")
    config = build_config(max_chunk_size=2000)
    chunks = chunk_python_content(PY_CONTENT, PY_PATH, config)
    print(f"chunk_python_content: {len(chunks)} chunks")

    print("\nA file that fails to parse returns [] instead of raising:")
    broken = chunk_python_content("def broken(:\n  pass", "broken.py", config)
    print(f"  chunk_python_content(<invalid syntax>) -> {broken""}")

    print(
        "\nchunk_python_file wraps the same thing with a plain max_chunk_size:"
    )
    chunks2 = chunk_python_source(PY_CONTENT, PY_PATH, max_chunk_size=2000)
    print(f"  chunk_python_source(...): {len(chunks2)} chunks (same result)")


def demo_chunk_python_files() -> None:
    title("chunk_python_files -- loops chunk_python_content over the corpus")
    chunks = chunk_python_files(CORPUS_ROOT, max_chunk_size=2000)
    print(f"{len(chunks)} python chunks across the whole corpus.")


def demo_chunk_corpus() -> None:
    title("chunk_corpus -- markdown chunks + python chunks, combined")
    chunks = chunk_corpus(CORPUS_ROOT, max_chunk_size=2000)
    md = sum(1 for c in chunks if c.kind == "Markdown")
    py = len(chunks) - md
    print(
        f"chunk_corpus(): {len(chunks)} chunks total"
        f" ({md} markdown, {py} python)"
    )
    print("This is what src/__main__.py's index command should call.")


if __name__ == "__main__":
    set_trace()
    demo_build_config()
    demo_split_span()

    demo_iter_corpus_files()
    demo_to_corpus_path()

    demo_parse_headers()
    demo_build_header_path()
    demo_split_markdown_section()
    demo_merge_small_sections()
    demo_build_chunk()
    demo_chunk_markdown_content()
    demo_chunk_markdown_files()

    demo_build_line_starts()
    demo_find_comment_start()
    demo_build_node_span()
    demo_parse_units_and_parse_python_units()
    demo_build_python_chunks()
    demo_chunk_python_content_and_source()
    demo_chunk_python_files()

    demo_chunk_corpus()
