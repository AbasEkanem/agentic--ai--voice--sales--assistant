"""Enterprise knowledge-base retrieval pipeline (FAISS + local embeddings).

Design notes / fixes over the first version:

- No work at import time. The embedding model and the FAISS index are built on the
  first call to get_vector_store(), not while the process boots. The old code ran
  the whole load inside the import chain (agent_tools -> graph -> main), which meant
  a model download and index build during LiveKit server startup.
- Correct loader per file type. .docx files are read with Docx2txtLoader; PDFs with
  PyPDFLoader. The old code fed a .docx (and later a bare directory) to PyPDFLoader,
  which cannot parse either.
- Matching index names. We always persist and load with the SAME index_name, so the
  second run actually finds the index it saved. The old code saved with the default
  name ("index") but loaded expecting "Knowledge_base".
- Configurable source. The document path comes from RAG_SOURCE (a file) or
  RAG_SOURCE_DIR (a folder of mixed files), so it is no longer tied to one machine.

The public surface for the rest of the app is `get_vector_store()`. A module-level
`__getattr__` (PEP 562) also exposes the old `vector_store` name lazily, so existing
imports keep working without triggering any work at import time.

Env: RAG_INDEX_DIR (default "ENTERPRISE_KNOWLEDGE_BASE"), RAG_SOURCE,
     RAG_SOURCE_DIR, RAG_EMBED_MODEL (default sentence-transformers/all-MiniLM-L6-v2).

RAG_SOURCE may be a local path OR an http(s) URL. A URL is downloaded once and cached
under RAG_CACHE_DIR (default ".rag_cache"); set RAG_REFETCH=1 to force a re-download.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger("rag_pipeline")

INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "ENTERPRISE_KNOWLEDGE_BASE")
INDEX_NAME = "Knowledge_base"  # used for BOTH save_local and load_local
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CACHE_DIR = Path(os.environ.get("RAG_CACHE_DIR", ".rag_cache"))
DOWNLOAD_TIMEOUT = float(os.environ.get("RAG_DOWNLOAD_TIMEOUT", "60"))

# Fallback kept for convenience; prefer RAG_SOURCE / RAG_SOURCE_DIR in the environment.
DEFAULT_SOURCE = r"C:\Users\Bussiness Sensor\Desktop\Apple_US_Education_Institution_Price_List - Copy.pdf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_store = None  # lazily-built cache; reset via reload()


def _embedding_model():
    # Imported lazily: langchain_huggingface pulls in torch, which is slow to import
    # and must not happen on the server's boot path.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})


def _loader_for(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix == ".docx":
        return Docx2txtLoader(str(path))
    return None


def _download_to_cache(url: str) -> Path:
    """Download a source document once and cache it under CACHE_DIR.

    Re-downloads only when the file is missing or RAG_REFETCH=1. The filename is
    taken from the URL path (falling back to a stable hash) so the extension is
    preserved for loader selection.
    """
    import hashlib
    from urllib.parse import unquote, urlparse

    from urllib.request import urlopen

    name = Path(unquote(urlparse(url).path)).name or "download"
    if "." not in name:  # no extension in the URL — keep the bytes uniquely keyed
        name = f"{hashlib.sha256(url.encode()).hexdigest()[:16]}-{name}"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / name
    if dest.exists() and os.environ.get("RAG_REFETCH") != "1":
        log.info("using cached source %s", dest)
        return dest

    log.info("downloading knowledge-base source from %s", url)
    try:
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT) as resp:  # noqa: S310 - user-provided URL
            dest.write_bytes(resp.read())
    except Exception as exc:
        if dest.exists():
            log.warning("download failed (%s); using stale cache %s", exc, dest)
            return dest
        raise RuntimeError(f"Could not download knowledge-base source {url}: {exc}") from exc
    log.info("saved source to %s", dest)
    return dest


def _load_documents():
    """Load the source file(s), choosing a loader per extension.

    RAG_SOURCE may be a local path or one/more http(s) URLs (comma- or newline-
    separated). URLs are downloaded and cached; local paths are read directly.
    RAG_SOURCE_DIR points at a local folder of mixed files.
    """
    source_dir = os.environ.get("RAG_SOURCE_DIR")
    if source_dir:
        base = Path(source_dir)
        if not base.is_dir():
            raise FileNotFoundError(f"RAG_SOURCE_DIR is not a directory: {base}")
        files = [p for p in sorted(base.iterdir()) if p.is_file()]
    else:
        raw = os.environ.get("RAG_SOURCE", DEFAULT_SOURCE)
        entries = [e.strip() for e in raw.replace("\n", ",").split(",") if e.strip()]
        if not entries:
            raise RuntimeError("RAG_SOURCE is empty.")
        files = []
        for entry in entries:
            if entry.lower().startswith(("http://", "https://")):
                files.append(_download_to_cache(entry))
            else:
                path = Path(entry)
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Knowledge-base source not found: {path}. "
                        "Set RAG_SOURCE to a file path or URL (or RAG_SOURCE_DIR to a folder)."
                    )
                files.append(path)

    docs = []
    for f in files:
        loader = _loader_for(f)
        if loader is None:
            log.warning("skipping unsupported knowledge-base file: %s", f)
            continue
        loaded = loader.load()
        docs.extend(loaded)
        log.info("loaded %d document(s) from %s", len(loaded), f)
    return docs


def _build_vector_store():
    from langchain_community.vectorstores import FAISS

    embeddings = _embedding_model()
    index_dir = Path(INDEX_DIR)

    if (index_dir / f"{INDEX_NAME}.faiss").exists():
        log.info("loading existing knowledge-base index from %s", index_dir)
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True,
        )

    log.info("no index at %s; building from source", index_dir)
    documents = _load_documents()
    if not documents:
        raise RuntimeError("No documents were loaded for the knowledge base.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    store = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir), index_name=INDEX_NAME)
    log.info("saved knowledge-base index (%d chunks) to %s", len(chunks), index_dir)
    return store


def get_vector_store():
    """Return the FAISS store, building or loading it once on first use."""
    global _store
    if _store is None:
        _store = _build_vector_store()
    return _store


def reload():
    """Drop the cached store (call after re-indexing the knowledge base)."""
    global _store
    _store = None


def __getattr__(name: str):
    # Backwards compatibility: `from RAG_pipeline import vector_store` still works,
    # but the store is only built when actually used (PEP 562 module __getattr__).
    if name == "vector_store":
        return get_vector_store()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_vector_store", "reload", "vector_store"]
