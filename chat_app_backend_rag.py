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


def _generate_relavent_chunks(query: str, results) -> tuple[list, bool]:
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

    return relevant_docs, output.relevant


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

    relevant_docs, is_relevant = _generate_relavent_chunks(query, all_results)

    if not is_relevant or not relevant_docs:
        return 'No relevant documents were found for this query in the uploaded files. Do not retry the search — answer based on general knowledge or inform the user.'

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