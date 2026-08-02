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

    def postings(self, term: str) -> tuple[Array, Array]:
        """
        Documents containing 'term' and their term frequencies
        Term X occurs in document Y, tf times.
        """
        term_id = self.vocab.get(term, None)
        if term_id is None:
            # if it doesn't exist we return an empty tuple
            empty = np.empty(0, dtype=np.int32)
            return empty, empty
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
    n_terms, n_docs = len(vocab), len(doc_len)
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

corpus: list[Chunk] = chunk_corpus()
short_corpus: list[Chunk] = [x for x in corpus[:2]]
index = build_index(short_corpus)
#terms = sorted(index.vocab, key=lambda term: index.vocab[term])
terms = [term for term in index.vocab]
terms_sorted = sorted(index.vocab, key=lambda term: index.vocab[term])

print(f"Normal: \n\n\n\n{index.vocab}\n\n")
print(f"Sorted: {terms_sorted}\n\n\n")
print(f"Not Sorted: {terms}")
print(terms_sorted == terms)


def save_index(index: Index, directory: str | Path) -> None:
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
    #terms = sorted(index.vocab, key=lambda term: index.vocab[term])
    #terms = [""] * len(index.vocab)
    #for term, term_id in index.vocab.items():
    #    terms[term_id] = term
    terms = list(index.vocab)
    (path / "vocab.json").write_text(json.dumps(terms), encoding="utf-8")
    (path / "sources.json").write_text(json.dumps([src.model_dump() for src in index.sources]), encoding="utf-8")



"""
print("All the postings, in the order they were collected")

inv = {v: k for k, v in vocab.items()}
print(f"term term_id doc tf\n")

for ti, d, tf in zip(term_ids, doc_ids, tfs):
    print(f"{inv[ti]} {ti} {d} {tf}")

print(f"{len(term_ids)} postings")

print("Sort by term_id, so each term rows are together")

terms = np.asarray(term_ids)
order = np.argsort(terms, kind="stable")
doc_ids = np.asarray(doc_ids)[order]
tfs = np.asarray(tfs)[order]
df = np.bincount(terms, minlength=len(vocab))
idxptr = np.zeros(len(vocab) + 1, dtype=np.int64)
np.cumsum(df, out=idxptr[1:])

print(f"row term term_id doc tf\n")

for row, (ti, d, tf) in enumerate(zip(terms[order], doc_ids, tfs)):
    mark = " <-- created" if row in idxptr[:-1] else ""
    print(f"{row} {inv[ti]} {ti} {d} {tf}{mark}")

print()
print("let the term column, stick with the boundaries")

print(f"doc_ids: {list(doc_ids)}")
print(f"tfs: {list(tfs)}")
print(f"idxptr: {list(idxptr)}")
print(f"vocab: {vocab}")
print()

print("Reading it again:")

for term, tid in vocab.items():
    s, e = idxptr[tid], idxptr[tid + 1]
    print(
        f"{term} rows {s}..{e-1} docs{list(doc_ids[s:e])}"
        f"\ttf={list(tfs[s:e])} df={e-s}"
    )
"""
