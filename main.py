import os
import warnings
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


from pathlib import Path
import time
import asyncio

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessageChunk

from chat_app_backend import chat, get_summary_for_chatHead
from chat_app_backend_rag import add_documents_to_store, clear_collection

# Use absolute paths derived from this file so the app works regardless of the
# directory uvicorn is started from (relative paths were breaking upload/static).

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / ".uploaded_files"
UPLOAD_DIR.mkdir(exist_ok=True)

NODE_LABELS = {
    "chat_message": "Thinking",
    "tools": "Using tools",
}

app = FastAPI()


@app.middleware("http")
async def no_cache_for_static(request, call_next):
    """Prevent the browser from caching the frontend files — otherwise a stale
    app.js/index.html gets reused and can send requests that don't match the
    current API (e.g. POST /chat 422)."""
    response = await call_next(request)
    if request.url.path in ("/", "") or request.url.path.endswith((".html", ".css", ".js", ".svg")):
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# Chat streaming (SSE)
# ---------------------------------------------------------------------------

def _sse(event: str, data: str) -> str:
    """Format one Server-Sent Event. Newlines in `data` must be escaped —
    SSE treats a bare newline as the end of the data field."""
    safe = data.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    return f"event: {event}\ndata: {safe}\n\n"

async def stream_chat_response(text: str, thread_id: str, kb_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    buffer = ""          # accumulates the CURRENT attempt's answer — never sent live
    final_answer = None  # set once check_answer approves (or gives up after retries)
    current_id = None
    has_tool_call = False
    last_label = None

    async for kind, payload in chat.astream(
        {"message": [HumanMessage(text)], "kb_id": kb_id},
        config=config,
        stream_mode=["messages", "updates"],
    ):
        if kind != "messages":
            # A node just finished — use it to detect retries / approval, and to
            # surface phase text during the silent generation/check gaps.
            for node_name, state_update in payload.items():
                msgs = state_update.get("message") or []
                last = msgs[-1] if msgs else None

                if node_name == "chat_message" and last is not None and not getattr(last, "tool_calls", None):
                    yield _sse("phase", "Checking your answer")

                elif node_name == "check_answer":
                    if last is not None and last.__class__.__name__ == "HumanMessage":
                        # Judge rejected it — discard this attempt's buffer entirely.
                        buffer = ""
                        yield _sse("phase", "Regenerating a better answer")
                    elif last is not None and last.__class__.__name__ == "AIMessage":
                        # Retries exhausted — this replacement message IS the final answer.
                        final_answer = last.content
                    else:
                        # Empty message list — judge approved the current buffer as-is.
                        final_answer = buffer
            continue

        # kind == "messages": token-level chunks. Only chat_message chunks matter —
        # we just accumulate them silently, we never yield "token" here.
        message_chunk, metadata = payload
        node = metadata.get("langgraph_node")

        if node != "chat_message":
            continue
        if not isinstance(message_chunk, AIMessageChunk):
            continue

        if message_chunk.id != current_id:
            buffer = ""
            has_tool_call = False
            current_id = message_chunk.id

        if message_chunk.tool_call_chunks:
            has_tool_call = True
            label = "Using tools"
            if label != last_label:
                yield _sse("status", label)
                last_label = label
            continue

        if message_chunk.content and not has_tool_call:
            buffer += message_chunk.content
        elif last_label != "Thinking":
            yield _sse("status", "Thinking")
            last_label = "Thinking"

    # Graph fully finished. Stream out the approved answer with a simulated
    # typewriter effect (generation already happened — this is just presentation).
    text_to_send = final_answer if final_answer is not None else buffer
    for i in range(0, len(text_to_send), 12):
        yield _sse("token", text_to_send[i:i + 12])
        await asyncio.sleep(0.02)

    yield _sse("done", "")


@app.post("/chat")
async def chat_endpoint(
    text: str = Form(...),
    thread_id: str = Form(...),
    kb_id: str = Form(...),
):
    return StreamingResponse(
        stream_chat_response(text, thread_id, kb_id),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Thread history + titles
# ---------------------------------------------------------------------------

@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    snapshot = await chat.aget_state({"configurable": {"thread_id": thread_id}})
    messages = snapshot.values.get("message", []) if snapshot.values else []

    out = []
    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        # Skip tool/system messages — the frontend only renders user/assistant turns.
        if m.__class__.__name__ in ("SystemMessage", "ToolMessage"):
            continue
        content = m.content
        # Skip empty assistant entries (tool-call preambles) so no blank bubbles.
        if role == "assistant" and not content:
            continue
        out.append({"role": role, "content": content})

    return {"thread_id": thread_id, "messages": out}


@app.post("/threads/{thread_id}/title")
async def generate_title(thread_id: str, first_message: str = Form(...)):
    title = get_summary_for_chatHead(first_message)
    return {"thread_id": thread_id, "title": title}


# ---------------------------------------------------------------------------
# Document upload / indexing
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_files(kb_id: str = Form(...), files: list[UploadFile] = File(...)):
    filenames = [f.filename or "file" for f in files]
    saved_paths = []
    success = False
    error = None
    try:
        # Write uploads first (outside the indexing try so path problems are
        # reported as a clean error, not a 500).
        for f in files:
            target = UPLOAD_DIR / (f.filename or f"upload_{len(saved_paths)}")
            target.write_bytes(await f.read())
            saved_paths.append(str(target))

        add_documents_to_store(saved_paths, collection_name=kb_id)
        success = True
    except Exception as e:
        error = str(e)
        print(f"[upload] indexing failed for {filenames}: {type(e).__name__}: {e}")
    finally:
        for p in saved_paths:
            Path(p).unlink(missing_ok=True)

    return JSONResponse({
        "success": success,
        "error": error,
        "filenames": filenames,
    })


# ---------------------------------------------------------------------------
# Reset a client's RAG collection (called when a browser refreshes, so each
# session starts clean and users never share document histories).
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset_client(kb_id: str = Form(...)):
    try:
        clear_collection(kb_id)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

app.mount("/src", StaticFiles(directory=str(BASE_DIR / "src")), name="src")
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")