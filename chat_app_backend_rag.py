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



load_dotenv()
llm = ChatGroq(model='openai/gpt-oss-120b', temperature=0.2)
llm_structured = ChatGroq(model='openai/gpt-oss-120b', temperature=0.2, disable_streaming=True)


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




def generate_output(query: str, vector_store):
    # Fast path: one retrieval pass, no query-expansion LLM call (that extra
    # call doubled latency without improving results).
    results = vector_store.max_marginal_relevance_search(query=query, k=4)
    content = [doc.page_content for doc in results]
    metadatas = [doc.metadata for doc in results]

    if not content:
        return "No documents have been uploaded yet. Please upload a file to enable document-based answers."

    # Build a clean, de-duplicated context string with source labels and hand it
    # straight back to the main model. (No extra LLM call inside the tool — that
    # halved the previous latency; the main model composes the final answer.)
    seen = set()
    context_parts = []
    for meta, text in zip(metadatas, content):
        key = (text or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        source = (meta or {}).get("source", "document")
        context_parts.append(f"[Source: {source}]\n{key}")
    return "\n\n".join(context_parts)

# @tool
# def rag_tool(query : str):

#     """Search and retrieve relevant information from the user's uploaded documents 
#     or knowledge base.

#     Use this tool whenever the user asks a question that could be answered by 
#     specific facts, data, definitions, or content that may exist in their documents 
#     — including questions about people, projects, numbers, dates, or anything not 
#     considered common/general knowledge.

#     Do NOT use this tool for:
#     - Greetings or small talk (e.g. "hi", "how are you")
#     - Simple math or logic questions
#     - General knowledge the model already knows confidently
#     - Follow-up questions that are just clarifying tone/formatting, not facts

#     Args:
#         query: A clear, standalone search query representing what the user wants 
#         to find. Rephrase vague or pronoun-heavy user questions into a specific, 
#         self-contained query (e.g., convert "what about its pricing?" into 
#         "product pricing details").

#     Returns:
#         A string containing the most relevant retrieved passages, or a message 
#         indicating no relevant documents were found.
#     """

#     vector_store = add_documents_to_store(st.session_state['pdf_files'])
#     generated_output = generate_output(query, vector_store)

#     return generated_output





#  **************************** test **************************

# llm_with_tools = llm.bind_tools([rag_tool])

# message = [HumanMessage(content=st.session_state['message'][-1])]
# result = llm_with_tools.invoke(message)
# message.append(result)

# if result.tool_calls:
#     for tool_call in result.tool_calls:
#         if tool_call["name"] == "rag_tool":
#             output = rag_tool.invoke(tool_call["args"])
#         else:
#             output = f"Error: unknown tool '{tool_call['name']}'"
#         message.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))

#     final_result = llm.invoke(message)
#     print(final_result.content)
# else:
#     print(result.content)

# print(st.session_state['message'][-1])