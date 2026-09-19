import asyncio
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import warnings
import os
import re

from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
os.environ["USE_TF"] = "0"

load_dotenv()


# (query expansion + chunk relevance grading). A second, unused ChatGroq
# instance was previously created here and never called — removed.
llm_structured = ChatGroq(model='openai/gpt-oss-120b', temperature=0.0, disable_streaming=True, max_tokens=512)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ********************Embedding**********************
EMBED_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"





# # ------------------------------------------
# from pathlib import Path
# get_path = Path(r"D:\AI-ML\AI Projects\fastapi-chat-app\eval_files\mml-book.pdf")
# get_path = [str(get_path)]
# # ------------------------------------------





_embeddings_instance = None
_vectorstore_cache = {}


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


def clear_collection(collection_name: str = "file_embeddings") -> None:
    vs = _get_vectorstore(collection_name)
    vs.delete_collection()
    _vectorstore_cache.pop(collection_name, None)


def add_documents_to_store(file_paths: list[str],
                           collection_name: str = "file_embeddings"):
    chunks = _load_and_split(file_paths)
    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(chunk.metadata.get("source", ""))
    vector_store = _get_vectorstore(collection_name)
    vector_store.add_documents(chunks)
    return vector_store


def _load_and_split(file_paths: list[str]):
    all_docs = []
    for file in file_paths:
        ext = file.split(".")[-1].lower()
        if ext == "pdf":
            docs = PyPDFLoader(file).load()
        elif ext == "docx":
            docs = Docx2txtLoader(file).load()
        elif ext in ("txt", "md"):
            docs = TextLoader(file, encoding="utf-8").load()
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
        return True
    except Exception as e:
        print(f"Error deleting {file_paths}: {e}")
        return False


class QueryExpansion(BaseModel):
    queries: list[str] = Field(
        description="1 more rephrasings or related variations of the original query, "
                    "covering different phrasings, synonyms, or angles the user might mean. "
                    "Do not include the original query itself."
    )


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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())


def rerank_chunks(query: str, results, top_k: int = 3) -> tuple[list, bool]:
    """Rerank retrieved chunks and keep the top three positive matches."""
    if not results:
        return [], False

    tokenized_chunks = [_tokenize(doc.page_content or "") for doc in results]
    query_tokens = _tokenize(query)
    if not query_tokens or not any(tokenized_chunks):
        return [], False

    scores = BM25Okapi(tokenized_chunks).get_scores(query_tokens)
    ranked_indices = sorted(range(len(results)), key=lambda index: scores[index], reverse=True)
    relevant_indices = [index for index in ranked_indices if scores[index] > 0][:top_k]
    print([results[index].page_content for index in relevant_indices])
    return [results[index] for index in relevant_indices], bool(relevant_indices)


async def _mmr_search_all(vector_store, queries: list[str], k: int = 4):
    """Run MMR search for every expanded query concurrently instead of one
    at a time — this is the main latency win in the RAG pipeline."""
    try:
        tasks = [
            vector_store.amax_marginal_relevance_search(query=q, k=k)
            for q in queries
        ]
        return await asyncio.gather(*tasks)
    except (AttributeError, NotImplementedError):
        # Installed langchain-chroma version has no native async MMR search —
        # fall back to running the sync calls concurrently in a thread pool.
        tasks = [
            asyncio.to_thread(vector_store.max_marginal_relevance_search, query=q, k=k)
            for q in queries
        ]
        return await asyncio.gather(*tasks)


async def generate_output(query: str, vector_store):

    expanded_queries = _expand_query(query)

    results_per_query = await _mmr_search_all(vector_store, expanded_queries, k=4)

    seen_ids = set()
    all_results = []
    for results in results_per_query:
        for doc in results:
            key = doc.page_content.strip()
            if key not in seen_ids:
                seen_ids.add(key)
                all_results.append(doc)

    relevant_docs, is_relevant = rerank_chunks(query, all_results, top_k=3)

    if not is_relevant or not relevant_docs:
        return 'No related chunks were found for this query in the uploaded files. Do not retry the search — answer based on general knowledge or inform the user.'

    seen = set()
    context_parts = []
    for doc in relevant_docs:
        key = (doc.page_content or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        source = (doc.metadata or {}).get("source", "document")
        context_parts.append(f"[Source: {source}]\n{key}")

    return "\n\n".join(context_parts)