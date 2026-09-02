# Personal AI Assistant — Project Documentation

## What Is This Project?

A full-stack, browser-based AI chat application built with **FastAPI** on the backend and **vanilla HTML/CSS/JS** on the frontend. It provides a conversational AI assistant that can:

- Hold multi-turn conversations using persistent, per-browser chat threads
- Search the web for real-time or factual information
- Answer questions from user-uploaded documents (PDF, DOCX, TXT, MD) using RAG
- Evaluate math expressions using a symbolic calculator
- Stream responses token-by-token for a real-time typewriter feel
- Verify its own answers with a self-correction loop via a LangGraph `check_answer` node

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI + Uvicorn |
| **LLM Orchestration** | LangGraph (StateGraph), LangChain |
| **LLMs** | Groq-hosted models via `langchain-groq` (`gpt-oss-120b`, `gpt-oss-20b`, `qwen3.6-27b`) |
| **RAG Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | ChromaDB (persisted locally in `/vectorstore`) |
| **Chat Memory** | SQLite via `langgraph-checkpoint-sqlite` |
| **Document Loaders** | PyPDF, Docx2txt, TextLoader (LangChain Community) |
| **Web Search** | DuckDuckGo via `langchain-community` |
| **Math Evaluation** | SymPy |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript (ES6+) |
| **Markdown Rendering** | marked.js (with custom math + code extensions) |
| **Syntax Highlighting** | highlight.js |
| **Math Rendering** | KaTeX |
| **Fonts** | Google Fonts — Comfortaa, Newsreader, Inter |
| **Environment** | python-dotenv (`.env` file) |

---

## Project Structure

```
fastapi-chat-app/
├── main.py                    # FastAPI app — API routes, SSE streaming
├── chat_app_backend.py        # LangGraph graph definition + conversation logic
├── chat_app_backend_rag.py    # RAG pipeline: embeddings, Chroma, retrieval, grading
├── llm_router.py              # Model fallback chain (primary → secondary → tertiary)
├── prompts.py                 # System prompt (SYSTEM_PROMPT)
├── tools.py                   # Tool definitions: calculator, web_search, rag_tool
├── requirements.txt           # Python dependencies
├── .env                       # API keys (GROQ_API_KEY, etc.) — not committed
├── chat_history.sqlite        # Persistent conversation memory (LangGraph checkpointer)
├── vectorstore/               # Persisted ChromaDB collections
├── .hf_cache/                 # Local HuggingFace embedding model cache
├── .uploaded_files/           # Temporary file storage for RAG indexing
├── retrive_docs/              # Debug output: retrieved chunks before/after grading
├── static/
│   ├── index.html             # Main HTML shell
│   ├── style.css              # All CSS — layout, components, typography, animations
│   └── app.js                 # All frontend JS — state, SSE streaming, UI logic
└── src/                       # Source assets (SVG logo, etc.)
```

---

## File-by-File Breakdown

### `main.py` — FastAPI Application Entry Point
- Mounts `static/` folder as the frontend
- Defines all HTTP endpoints:
  - `POST /chat` → starts streaming response via SSE
  - `GET /threads/{thread_id}` → fetches conversation history for a thread
  - `POST /threads/{thread_id}/title` → auto-generates a short thread title
  - `POST /upload` → receives files, saves them, indexes them into ChromaDB
  - `POST /reset` → clears a browser's RAG collection
- Implements `_sse(event, data)` helper to correctly escape newlines in SSE payloads
- Implements `stream_chat_response()` — async generator that streams LangGraph output as SSE events (`token`, `status`, `clear`, `done`)
- Adds a no-cache middleware for all frontend static assets

---

### `chat_app_backend.py` — LangGraph Conversation Graph
Defines the stateful conversation workflow as a **LangGraph StateGraph**.

**State schema (`MessageState`)**
- `message` — accumulated list of LangChain messages (`add_messages` reducer)
- `kb_id` — the browser's unique knowledge base ID (per-session RAG namespace)
- `retry_count` — how many self-correction retries have been attempted

**Nodes**
| Node | Role |
|---|---|
| `chat_message` | Calls the LLM (with tools bound) via `invoke_with_fallback` |
| `tools` | LangChain ToolNode — executes tool calls returned by the LLM |
| `check_answer` | Grades the final answer against the original question; optionally replaces it |

**Graph flow**
```
START → chat_message → [tools_condition]
    ├─ if tool call → tools → chat_message (loop)
    └─ if no tool call → check_answer
         ├─ if relevant → END
         └─ if not relevant & retries < 2 → chat_message (retry)
              └─ if retries exhausted → END (fallback message)
```

Also contains `get_summary_for_chatHead(user_message)` — generates a short 5-word thread title.

---

### `chat_app_backend_rag.py` — RAG Pipeline
Handles all document ingestion and retrieval.

**Embedding**
- Model: `sentence-transformers/all-MiniLM-L6-v2` (loaded once, singleton)
- Cached locally in `.hf_cache/`

**Vector Store**
- ChromaDB, persisted to `/vectorstore`
- One collection per `kb_id` (one per browser session)
- `_get_vectorstore(collection_name)` — lazily creates / returns cached instance
- `clear_collection(collection_name)` — wipes a session's documents

**Document Ingestion**
- `add_documents_to_store(file_paths, collection_name)` — loads files, splits, embeds, stores
- `_load_and_split(file_paths)` — supports `.pdf`, `.docx`, `.txt`, `.md`
- Chunk size: 800 tokens, 150 overlap

**RAG Retrieval Pipeline (`generate_output`)**
1. **Query Expansion** — LLM generates 2 alternate phrasings of the user query (structured output: `QueryExpansion`)
2. **MMR Search** — runs each expanded query with `max_marginal_relevance_search(k=3)` to reduce redundancy
3. **Chunk Grading** — LLM grades all retrieved chunks for relevance (structured output: `check_chunk_quality`), returning only relevant indices
4. **Context Assembly** — returns formatted `[Source: filename]\n<content>` strings for the LLM

---

### `llm_router.py` — Model Fallback Chain
Defines an ordered list of Groq-hosted LLMs. On `RateLimitError`, falls back to the next model:

1. `openai/gpt-oss-120b` (primary, most capable)
2. `openai/gpt-oss-20b` (secondary)
3. `qwen/qwen3.6-27b` (tertiary fallback)

`invoke_with_fallback(messages, tools)` wraps tool binding and model invocation.

---

### `prompts.py` — System Prompt
Defines `SYSTEM_PROMPT` as a LangChain `SystemMessage`. Instructs the LLM to:
- Behave as a warm, teacher-like mentor
- Use emojis tastefully (not on every sentence)
- End each response with a bold follow-up question
- Use the calculator for all math
- Use document search for uploaded file queries
- Use web search for real-time or external facts
- Keep formatting clean — no multiple consecutive blank lines
- Avoid LaTeX notation in math

---

### `tools.py` — Tool Definitions
Three LangChain `@tool` functions bound to the LLM:

| Tool | Description |
|---|---|
| `web_search(query)` | DuckDuckGo search for real-time or general facts |
| `calculator(expression)` | SymPy symbolic evaluator for math expressions |
| `rag_tool(query, state)` | Retrieves from the user's ChromaDB collection using `kb_id` injected from LangGraph state |

---

### `static/index.html` — Frontend Shell
Single-page HTML with:
- Sidebar: app title ("AI Assistant"), New Chat button, collapse toggle, thread list
- Top brand header (`#brand`): always-visible logo + "Personal AI Assistant" title
- Sidebar expand button (`#sidebar-expand`): appears when sidebar is collapsed
- Messages area: scrollable container with hero empty state
- Composer: text input, file attach button, file chips, send button
- CDN script imports: marked.js, KaTeX, highlight.js

---

### `static/app.js` — Frontend Logic
All client-side JavaScript.

**State (localStorage)**
- `kb_id` — unique UUID per browser, scopes the RAG collection
- `threads` — JSON array of `{ thread_id, title }` objects
- `active_thread_id` — currently selected thread

**Key Functions**
| Function | Role |
|---|---|
| `renderBubble(el, text)` | Parses markdown with `marked.js`; sanitizes trailing spaces and excess newlines |
| `appendMessage(role, content)` | Creates a message DOM node and inserts it |
| `appendUserMessage(text, filenames, error)` | Creates user message with optional file attachment cards |
| `loadThread(threadId)` | Fetches thread history from `GET /threads/{id}` and renders all messages |
| `pruneStaleThreads()` | Removes threads with no server-side messages (e.g. after server restart) |
| `setSidebar(collapsed)` | Toggles `.app.collapsed`, shows/hides `»` expand button |
| `updateEmptyState()` | Manages hero state visibility, composer position |
| `renderThreadList()` | Renders sidebar thread list with active highlight and rename support |
| `startRename(thread, row)` | Inline thread rename with Enter/Escape/blur commit |
| `form submit handler` | Handles file upload → `/upload`, then chat → `/chat` SSE streaming |

**SSE Streaming**
Reads the SSE stream from `POST /chat` and handles four event types:
- `token` — appends to `answer` string, re-renders bubble via `renderBubble`
- `status` — shows "Thinking…" / "Using tools…" label
- `clear` — resets bubble for `check_answer` correction
- `done` — stream complete

**Markdown Extensions (marked.js)**
- `mathBlock` — block `$$ ... $$` rendered with KaTeX
- `mathInline` — inline `$...$` rendered with KaTeX
- `codeRenderer` — dark IDE-style code blocks with language label + copy button

---

### `static/style.css` — All Styling
Organized into sections:
- **CSS custom properties** (`:root`) — accent colour, backgrounds, borders, text colours
- **Body / global** — `Comfortaa`, `Newsreader`, `Inter` font stack applied globally
- **Sidebar** — width, collapse animation (`width: 0`, `margin-left: -268px`), thread rows, rename input
- **Top bar brand** — logo + "Personal AI Assistant", always visible, padding shifts when sidebar collapses
- **Messages** — user bubble (right-aligned), assistant bubble (left-aligned with avatar icon)
- **Markdown content** — paragraphs (0.8rem bottom margin), headings, lists (0.85rem bottom margin), code, tables, blockquotes
- **IDE code blocks** — dark `#282c34` background, language label, hover copy button
- **Bold text** — `Inter` font, `font-weight: 650`, high-contrast `#1e293b` color for `<strong>` and `<b>`
- **Composer** — rounded input bar, file attach, send button
- **File chips** — pending attachment pills above composer
- **Animations** — `pulse` for status label, hero leave animation (scale + translate + opacity)
- **Responsive** — `@media (max-width: 640px)` full-width layout

---

## Data Flow

### Chat Message (No Documents)
```
User types message
  → POST /chat (text, thread_id, kb_id)
    → stream_chat_response()
      → chat.astream() [LangGraph]
        → chat_message node
          → invoke_with_fallback(messages, tools)
            → [tool call] → tools node → back to chat_message
            → [answer] → check_answer node
              → _check_answer(question, answer)
                → relevant → SSE "done"
                → not relevant & retries < 2 → retry chat_message
  ← SSE "token" events
    → renderBubble() re-renders markdown incrementally in browser
```

### Document Upload + RAG Query
```
User attaches file(s) + question
  → POST /upload (kb_id, files[])
    → _load_and_split()         # load + chunk into 800-token pieces
    → add_documents_to_store()  # embed (MiniLM-L6-v2) + store in ChromaDB[kb_id]
  → POST /chat ("Attached files: ..." text)
    → LLM calls rag_tool(query, state)
      → _expand_query(query)              # 3 query variants via LLM
      → MMR search × 3 variants          # max_marginal_relevance_search(k=3)
      → _generate_relavent_chunks()      # LLM grades chunks, returns relevant indices
      → returns formatted [Source] context
    → LLM generates answer from context
    → check_answer validates
  ← SSE token stream → browser renders answer
```

### Thread Title Generation
```
First message sent in a new thread
  → POST /threads/{thread_id}/title (first_message)
    → get_summary_for_chatHead()   # single LLM call, no tools, ≤5 words
    → returns { title: "..." }
  → updateThreadTitle() → renderThreadList()  # sidebar updates
```

---

## Environment Variables (`.env`)

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | API key for Groq-hosted LLM inference |

---

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Start the development server
uvicorn main:app --reload
```

Open `http://localhost:8000` in a browser.

---

## Key Design Decisions

- **Per-browser isolation** — Each browser gets its own `kb_id` (UUID in localStorage) namespacing its ChromaDB collection. Documents uploaded in one browser are never visible to another.
- **Thread persistence** — Conversations survive server restarts via SQLite (LangGraph checkpointer). Thread IDs live in localStorage per browser.
- **LLM fallback chain** — Rate-limit errors transparently fall back through the model chain without user-facing failures.
- **RAG quality gate** — Retrieved chunks are LLM-graded before use as context, preventing weakly-relevant content from contaminating answers.
- **Self-correction loop** — `check_answer` validates LLM responses and can trigger retries with feedback, capped at 2 retries to prevent infinite loops.
- **SSE streaming** — Token-by-token streaming via Server-Sent Events; newlines in data payloads are escaped (`\n` → `\\n`) on the server and unescaped in JS.
- **Stale thread pruning** — On load, threads with no server-side messages are silently removed from the sidebar to prevent empty shell accumulation.
- **Text sanitization before rendering** — Before passing LLM output to `marked.parse()`, the JS client strips trailing whitespace from line ends and collapses 3+ consecutive newlines to prevent spurious blank lines in rendered markdown.
