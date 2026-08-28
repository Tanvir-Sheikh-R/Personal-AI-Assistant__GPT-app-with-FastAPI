from pathlib import Path
import time

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
    safe = data.replace("\n", "\\n")
    return f"event: {event}\ndata: {safe}\n\n"


async def stream_chat_response(text: str, thread_id: str, kb_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    buffer = ""
    current_id = None
    has_tool_call = False
    last_label = None
    correction = False  # True once a check_answer replacement starts streaming
    last_flush = time.monotonic()

    def should_flush():
        # Stream incrementally: flush as soon as enough characters accumulated
        # or enough time passed, so the client sees a smooth typewriter effect
        # without a per-token event storm.
        return bool(buffer) and (len(buffer) >= 12 or time.monotonic() - last_flush >= 0.04)

    async for message_chunk, metadata in chat.astream(
        {"message": [HumanMessage(text)], "kb_id": kb_id},
        config=config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node")
        label = NODE_LABELS.get(node, "Thinking")

        # The check_answer node may append a replacement answer after the first
        # one was already streamed. Flush the original, then tell the client to
        # clear the bubble before streaming the corrected answer.
        if node == "check_answer":
            if not isinstance(message_chunk, AIMessageChunk) or not message_chunk.content:
                continue
            if buffer and not has_tool_call:
                yield _sse("token", buffer)
                buffer = ""
                has_tool_call = False
            if not correction:
                correction = True
                yield _sse("clear", "")
            yield _sse("token", message_chunk.content)
            continue

        # Non-answer nodes (e.g. the ToolNode) produce a chunk per token, which
        # would otherwise flood the client with repeated "status" events. Emit
        # a status event only when the active phase actually changes.
        if node != "chat_message":
            if label != last_label:
                yield _sse("status", label)
                last_label = label
            continue

        if not isinstance(message_chunk, AIMessageChunk):
            continue

        if message_chunk.id != current_id:
            # A new message group started — flush the previous one, but only if
            # it turned out NOT to be a tool call (tool-call preambles are discarded).
            if buffer and not has_tool_call:
                yield _sse("token", buffer)
            buffer = ""
            has_tool_call = False
            current_id = message_chunk.id
            last_flush = time.monotonic()

        if message_chunk.tool_call_chunks:
            # This group is a tool call — its buffered content (if any) is never shown.
            has_tool_call = True

        if message_chunk.content:
            buffer += message_chunk.content
            if not has_tool_call and should_flush():
                yield _sse("token", buffer)
                buffer = ""
                last_flush = time.monotonic()
        elif label != last_label:
            # Still "Thinking" inside the answer node before any content arrives.
            yield _sse("status", label)
            last_label = label

    # Flush whatever's left once the stream ends (the final answer's group).
    if buffer and not has_tool_call:
        yield _sse("token", buffer)

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

app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")


# hi