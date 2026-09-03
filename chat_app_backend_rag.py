
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from llm_router import invoke_with_fallback 
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
# print('after worning')

load_dotenv()
llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.2)
llm_structured = ChatGroq(model='openai/gpt-oss-120b', temperature=0.1, disable_streaming=True)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ********************Embedding**********************
EMBED_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings_instance = None
_vectorstore_cache = {}


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            cache_folder=EMBED_CACHE
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
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

    
# ------------- check retrived chunks ---------------
def save_docs(query, results, suffix=""):
    os.makedirs('retrive_docs', exist_ok=True)

    with open(f'retrive_docs/{query}_{suffix}.txt', 'a', encoding='utf-8') as f:
        content = "\n\n".join(doc.page_content for doc in results)
        f.write(content)
# ---------------------------------------------------


class QueryExpansion(BaseModel):
    queries: list[str] = Field(
        description="2 more rephrasings or related variations of the original query, "
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
            Generate 2 alternative phrasings of the user's question to improve document retrieval.
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

    expanded_queries = _expand_query(query)

    seen_ids = set()
    all_results = []
    for q in expanded_queries:
        for doc in vector_store.max_marginal_relevance_search(query=q, k=3):
            key = doc.page_content.strip()
            if key not in seen_ids:
                seen_ids.add(key)
                all_results.append(doc)

    relevant_docs, is_relevant = _generate_relavent_chunks(query, all_results)

    # save_docs(query, relevant_docs, "after")        # -> remove this after checking
    # save_docs(query, all_results, "before")         # -> remove this after checking


    if not is_relevant or not relevant_docs:
        return 'No relevant documents found for the query. Please rephrase your question or upload relevant documents.'


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