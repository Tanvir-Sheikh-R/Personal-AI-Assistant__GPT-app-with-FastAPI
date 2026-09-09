from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_community.retrievers import BM25Retriever
from llm_router import invoke_with_fallback
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
os.environ["USE_TF"] = "0"

load_dotenv()
llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.2)
llm_structured = ChatGroq(model='openai/gpt-oss-120b', temperature=0.0, disable_streaming=True, max_tokens=512)


os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# ********************Embedding + Reranker**********************
EMBED_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small, CPU-friendly cross-encoder

# Retrieval tuning knobs
DENSE_K = 2          # chunks pulled per expanded query via dense/MMR search
BM25_K = 2            # chunks pulled per expanded query via BM25 keyword search
RERANK_TOP_K = 2      # chunks kept after cross-encoder reranking, before LLM grading

_embeddings_instance = None
_reranker_instance = None
_vectorstore_cache = {}
_bm25_cache = {}


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            cache_folder=EMBED_CACHE,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
    return _embeddings_instance


def _get_reranker() -> CrossEncoder:
    """Lazily loads a small CPU cross-encoder used to re-score retrieved chunks
    against the exact user query. Loaded once per process and reused."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder(
            RERANK_MODEL,
            cache_folder=EMBED_CACHE,
            device="cpu",
        )
    return _reranker_instance


def _get_vectorstore(collection_name: str = "file_embeddings") -> Chroma:
    if collection_name not in _vectorstore_cache:
        VECTORSTORE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vectorstore"
        )

        _vectorstore_cache[collection_name] = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=_get_embeddings(),
            collection_name=collection_name,
            )

    return _vectorstore_cache[collection_name]


def _get_bm25_retriever(collection_name: str):
    """Builds (and caches) a keyword-based BM25 retriever from whatever is
    currently stored in the given Chroma collection. This adds an exact-match
    layer on top of dense/semantic search, so names, IDs, and specific terms
    that don't embed distinctively are still findable. Returns None if the
    collection is empty or unreadable — callers must handle that."""
    if collection_name in _bm25_cache:
        return _bm25_cache[collection_name]

    vector_store = _get_vectorstore(collection_name)
    try:
        raw = vector_store.get(include=["documents", "metadatas"])
    except Exception as e:
        print(f"[bm25] failed to read collection '{collection_name}': {e}")
        _bm25_cache[collection_name] = None
        return None

    texts = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    if not texts:
        _bm25_cache[collection_name] = None
        return None

    docs = [Document(page_content=t, metadata=(m or {})) for t, m in zip(texts, metadatas)]
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = BM25_K
    _bm25_cache[collection_name] = retriever
    return retriever


def _invalidate_bm25_cache(collection_name: str) -> None:
    _bm25_cache.pop(collection_name, None)


def clear_collection(collection_name: str = "file_embeddings") -> None:
    vs = _get_vectorstore(collection_name)
    vs.delete_collection()
    _vectorstore_cache.pop(collection_name, None)
    _invalidate_bm25_cache(collection_name)


def add_documents_to_store(file_paths: list[str],
                           collection_name: str = "file_embeddings"):
    chunks = _load_and_split(file_paths)
    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(chunk.metadata.get("source", ""))
    vector_store = _get_vectorstore(collection_name)
    vector_store.add_documents(chunks)
    _invalidate_bm25_cache(collection_name)  # rebuilt lazily on next query
    return vector_store


def _load_text_document(file_path: str) -> Document:
    """Read text files commonly produced by Windows and web editors."""
    raw = open(file_path, "rb").read()
    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252")
    for encoding in encodings:
        try:
            return Document(
                page_content=raw.decode(encoding),
                metadata={"source": file_path},
            )
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode text file '{os.path.basename(file_path)}'")


def _load_and_split(file_paths: list[str]):
    all_docs = []
    for file in file_paths:
        ext = file.split(".")[-1].lower()
        if ext == "pdf":
            docs = PyPDFLoader(file).load()
        elif ext == "docx":
            docs = Docx2txtLoader(file).load()
        elif ext in ("txt", "md"):
            docs = [_load_text_document(file)]
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported types: pdf, docx, txt, md")

        if not any((d.page_content or "").strip() for d in docs):
            raise ValueError(
                f"No readable text found in '{os.path.basename(file)}' — "
                "it may be a scanned/image-only document."
            )
        all_docs.extend(docs)

    total_chars = sum(len(d.page_content or "") for d in all_docs)

    # Scale chunk size with document length
    if total_chars < 3000:
        chunk_size, chunk_overlap = 400, 80     # short doc — smaller, precise chunks
    elif total_chars < 20000:
        chunk_size, chunk_overlap = 800, 150    # default
    else:
        chunk_size, chunk_overlap = 1200, 300   # long doc — bigger chunks, fewer total

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", "  ", " ", ""],
    )
    return splitter.split_documents(all_docs)



def delete_documents_from_store(file_paths: list[str],
                                collection_name: str = "file_embeddings"):
    try:
        vector_store = _get_vectorstore(collection_name)
        filenames = [os.path.basename(f) for f in file_paths]
        vector_store._collection.delete(where={"source": {"$in": filenames}})
        _invalidate_bm25_cache(collection_name)
        return True
    except Exception as e:
        print(f"Error deleting {file_paths}: {e}")
        return False


# ------------- check retrived chunks ---------------
def save_docs(query, results, suffix=""):
    os.makedirs('retrive_docs', exist_ok=True)

    with open(f'retrive_docs/{query}_{suffix}.txt', 'a', encoding='utf-8') as f:
        content = "\n\n".join(doc.page_content for doc in results)
        f.write(content)
# ---------------------------------------------------


class QueryExpansion(BaseModel):
    queries: list[str] = Field(
        description="1 more rephrasings or related variations of the original query, "
                    "covering different phrasings, synonyms, or angles the user might mean. "
                    "Do not include the original query itself."
    )

class check_chunk_quality(BaseModel):
    relevant_indeces: list[int] = Field(
        description="0-based indices into the retrieved chunks that are relevant to the question"
    )
    relevant: bool = Field(description="True if atleast one relevant answer is found in the retrieved chunks else False")



def _expand_query(query: str) -> list[str]:
    expand_prompt = PromptTemplate(
        template="""
            ## You must respond by calling the QueryExpansion tool with your one rephrasings — do not answer in plain text
            Generate 1 alternative phrasings of the user's question to improve document retrieval.
            Include synonyms, related terms, and different ways the answer might be phrased in a document.
            Keep each variation short and standalone.

            Original question: {query}
        """,
        input_variables=['query']
    )
    try:
        output = llm_structured.with_structured_output(QueryExpansion).invoke(
            expand_prompt.format(query=query)
        )
        return [query] + output.queries
    except Exception as e:
        print(f"[query expansion] failed, using original query only: {e}")
        return [query]


def _hybrid_retrieve(queries: list[str], vector_store: Chroma, collection_name: str):
    """Runs dense (MMR) search AND BM25 keyword search for each expanded query,
    then merges and de-duplicates the results by content. Combines semantic
    similarity with exact keyword matching so names, IDs, and specific terms
    aren't missed just because they don't embed distinctively."""
    bm25_retriever = _get_bm25_retriever(collection_name)

    seen = set()
    merged = []

    for q in queries:
        dense_hits = vector_store.max_marginal_relevance_search(query=q, k=DENSE_K)
        keyword_hits = bm25_retriever.invoke(q) if bm25_retriever is not None else []

        for doc in dense_hits + keyword_hits:
            key = (doc.page_content or "").strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(doc)

    return merged


def _rerank(query: str, docs: list) -> list:
    """Re-scores retrieved chunks against the exact original query using a
    cross-encoder, which judges query-passage relevance more precisely than
    embedding similarity alone. Narrows a larger candidate pool down to the
    strongest few before the (more expensive) LLM relevance grader runs."""
    if not docs:
        return []
    if len(docs) <= RERANK_TOP_K:
        return docs

    try:
        reranker = _get_reranker()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:RERANK_TOP_K]]
    except Exception as e:
        print(f"[rerank] failed, falling back to first {RERANK_TOP_K} candidates: {e}")
        return docs[:RERANK_TOP_K]


def _generate_relavent_chunks(query: str, results) -> check_chunk_quality:
    if not results:
        return [], False
    
    check_prompt = PromptTemplate(
        template="""
            You are a strict relevance grader for a retrieval-augmented generation system.
            Your job is to decide which retrieved chunks, if any, contain information that
            actually helps answer the user's question — not just chunks that share keywords
            or topic overlap with it.

            Question: {query}
            Retrieved Chunks: {results}

            Instructions:
            - A chunk is relevant only if it contains facts, data, or content that directly
            helps answer the question — partial relevance counts if the chunk contributes
            a meaningful piece of the answer, even if it doesn't fully answer it alone.
            - A chunk is NOT relevant if it merely mentions the same topic, entity, or
            keywords without actually addressing what is being asked.
            - Do not use outside knowledge to judge correctness — only judge whether the
            chunk's content is relevant to the question, not whether it is factually true.
            - Be strict: when in doubt, exclude a chunk rather than include a weak match.
            - Return the 0-based indices of every chunk you judge relevant, in the order
            they appear. If no chunks are relevant, return an empty list.
            - Set `relevent` to True only if at least one chunk was judged relevant.
        """,
        input_variables=['query', 'results']
    )

    numbered = "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(results))
    try:
        output = llm_structured.with_structured_output(check_chunk_quality).invoke(
            check_prompt.format(query=query, results=numbered)
        )
    except Exception as e:
        print(f"[grading] failed, treating as no-match: {e}")
        return [], False

    relevant_docs = [results[i] for i in output.relevant_indeces if 0 <= i < len(results)]
    print(f"check_chunk_quality output: {output}")

    return relevant_docs, output.relevant



def generate_output(query: str, vector_store):

    collection_name = vector_store._collection.name

    expanded_queries = _expand_query(query)

    all_results = _hybrid_retrieve(expanded_queries, vector_store, collection_name)
    reranked_results = _rerank(query, all_results)

    relevant_docs, is_relevant = _generate_relavent_chunks(query, reranked_results)

    # save_docs(query, relevant_docs, "after")        # -> remove this after checking
    # save_docs(query, all_results, "before")         # -> remove this after checking


    if not is_relevant or not relevant_docs:
        return 'No relevant documents were found for this query in the uploaded files. Do not retry the search — answer based on general knowledge or inform the user.'


    seen = set()
    context_parts = []
    for doc in relevant_docs:
        key = (doc.page_content or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        metadata = doc.metadata or {}
        source = metadata.get("source", "document")
        page = metadata.get("page")
        label = f"[Source: {source}, page {page + 1}]" if isinstance(page, int) else f"[Source: {source}]"
        context_parts.append(f"{label}\n{key}")

    return "\n\n".join(context_parts)