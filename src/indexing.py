from src.models import MinimalSource, Chunk
import json
from pathlib import Path
import numpy as np
import numpy.typing as npt
import re
from collections import Counter
from typing import NamedTuple, TypeAlias, Callable
from tqdm import tqdm
from src.chunking import chunk_corpus

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

Array: TypeAlias = npt.NDArray


class Index(NamedTuple):
    vocab: dict[str, int] # every different term in the corpus
    idxptr: Array # where each term's postings start and end
    doc_ids: Array
    tfs: Array # frequency of a term appears in one document
    idf: Array # how much a term is worth (rare = high)
    doc_len: Array # accumulate the length of each chunk
    avgdl: float # average document length
    sources: list[MinimalSource]

    @property
    def n_chunks(self) -> int:
        """Number of indexed chunks"""
        return len(self.doc_len)

    def postings(self, term_id: int) -> tuple[Array, Array]:
        """
        Documents containing 'term' and their term frequencies
        Term X occurs in document Y, tf times.
        """
        start, end = self.idxptr[term_id], self.idxptr[term_id + 1]
        return self.doc_ids[start:end], self.tfs[start:end]

def split_identifier(word: str) -> list[str]:
    return PARTS.findall(word)

def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in WORD_RE.findall(text):
        tokens.append(word.lower())
        parts = split_identifier(word)
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)
    return tokens

def build_index(
    chunks: list[Chunk],
    tokenizer: Callable[[str], list[str]] = tokenize
) -> Index:
    """build the inverted index over the chunks"""
    vocab: dict[str, int] = {}
    term_ids: list[int] = []
    doc_ids: list[int] = []
    tfs: list[int] = []
    doc_len = np.zeros(len(chunks), dtype=np.int32)
    sources: list[MinimalSource] = []

    for doc_id, chunk in enumerate(tqdm(chunks, desc="indexing")):
        tokens = tokenizer(chunk.index_text)
        doc_len[doc_id] = len(tokens)
        sources.append(
            MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
            )
        )
        for term, tf in Counter(tokens).items():
            term_ids.append(vocab.setdefault(term, len(vocab)))
            doc_ids.append(doc_id)
            tfs.append(tf)

    return pack_postings(vocab, term_ids, doc_ids, tfs, doc_len, sources)

def pack_postings(
    vocab: dict[str, int],
    term_ids: list[int],
    doc_ids: list[int],
    tfs: list[int],
    doc_len: Array,
    sources: list[MinimalSource],
) -> Index:
    """
    Group (term, document, tf) into CSR arrays

    sort them by term id puts each term's postings together, so idxptr
    only has to remember where each block starts. Using stable sort
    we keep doc_ids ascending inside a block

    df: with one posting per term-document pair, counting term ids counts
    documents. idf is derived from it here.
    """
    n_terms = len(vocab)
    n_docs = len(doc_len)
    terms = np.asarray(term_ids, dtype=np.int32)
    order = np.argsort(terms, kind="stable")

    df = np.bincount(terms, minlength=n_terms)
    idxptr = np.zeros(n_terms + 1, dtype=np.int64)
    np.cumsum(df, out=idxptr[1:])

    return Index(
        vocab = vocab,
        idxptr = idxptr,
        doc_ids = np.asarray(doc_ids, np.int32)[order],
        tfs = np.asarray(tfs, np.int32)[order],
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)),
        doc_len = doc_len,
        avgdl = float(doc_len.mean()) if n_docs else 0.0,
        sources=sources
    )

def save_index(index: Index, directory: str | Path) -> None:
    """
    Save index in directory as files

    Numeric arrays go to 'postings.npz' the terms to 'vocab.json'
    in term-id order, the document locations to 'sources.json'

    .npz is a zip archive holding one .npy per array each .npy
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(
        path / "postings.npz",
        idxptr=index.idxptr,
        doc_ids=index.doc_ids,
        tfs=index.tfs,
        idf=index.idf,
        doc_len=index.doc_len,
    )
    terms = list(index.vocab)
    (path / "vocab.json").write_text(json.dumps(terms), encoding="utf-8")
    (path / "sources.json").write_text(json.dumps([src.model_dump() for src in index.sources]), encoding="utf-8")

def load_index(directory: str | Path) -> Index:
    """Load an index written by save_index"""

    path = Path(directory)
    with np.load(path / "postings.npz") as arrays:
        idxptr = arrays["idxptr"]
        doc_ids = arrays["doc_ids"]
        tfs = arrays["tfs"]
        idf = arrays["idf"]
        doc_len = arrays["doc_len"]
    terms = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    raw = json.loads((path / "sources.json").read_text(encoding="utf-8"))
    return Index(
        vocab={term: i for i, term in enumerate(terms)},
        idxptr=idxptr,
        doc_ids=doc_ids,
        tfs=tfs,
        idf=idf,
        doc_len=doc_len,
        avgdl=float(doc_len.mean()) if len(doc_len) else 0.0,
        sources = [MinimalSource(**item) for item in raw]
    )

def bm25(
    index: Index,
    query: str,
    top_k: int = 10,
    k1: float = 1.2,
    b: float = 0.75,
    tokenizer: Callable[[str], list[str]] = tokenize,
) -> list[MinimalSource]:
    k = min(index.n_chunks, top_k)
    if not k:
        return []
    scores = np.zeros(index.n_chunks, np.float32)
    for term in tokenizer(query):
        term_id = index.vocab.get(term, None)
        if term_id is None:
            continue
        docs, tf = index.postings(term_id)
        dl = index.doc_len[docs]
        denom = tf + k1 * (1 - b + b * dl / index.avgdl)
        scores[docs] += index.idf[term_id] * tf * (k1 + 1) / denom
    best = np.argpartition(scores, -k)[-k:]
    top = best[np.argsort(scores[best])[::-1]]
    return [index.sources[i] for i in top]

if __name__ == "__main__":
    import sys

    processed = Path("data/processed")
    default = "How do I serve a LoRA adapter with the OpenAI server?"

    if (processed/"postings.npz").exists():
        index = load_index(processed)
    else:
        index = build_index(chunk_corpus())
        save_index(index, processed)

    query = " ".join(sys.argv[1:]) if sys.argv[1:] else default
    print(f"query: {query}")
    print(f"index: {index.n_chunks} chunks, {len(index.vocab)} terms\n")

    def preview(source, width = 160):
        text = Path(source.file_path).read_text(encoding="utf-8")
        chunk = text[source.first_character_index:source.last_character_index]
        return " ".join(chunk.split())[:width]

    def explain(
        index: Index,
        query: str,
        doc_id: int,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> list[tuple[str, float]]:
        rows: list[tuple[str, float, float]] = []
        for term in tokenizer(query):
            term_id = index.vocab.get(term, None)
            if term_id is None:
                continue
            docs, tf = index.postings(term_id)
            dl = index.doc_len[docs]
            denom = tf + k1 * (1 - b + b * dl / index.avgdl)
            scores = np.zeros(index.n_chunks, np.float32)
            scores[docs] = index.idf[term_id] * tf * (k1 + 1) / denom
            rows.append((term, float(index.idf[term_id]), float(scores[doc_id])))
        return sorted(rows, key=lambda row: -row[2])

    results = bm25(index, query, top_k=5)
    for rank, source in enumerate(results, start=1):
        print(f"{rank}: {source.file_path}"
              f"[{source.first_character_index}:"
              f"{source.last_character_index}]")
        print(f"{preview(source)}\n")

    doc_id = index.sources.index(results[0])

    for term, idf, score in explain(index, query, doc_id)[:10]:
        print(f"   {term} - idf {idf:.2f} - score {score:.2f}")
