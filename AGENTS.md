# AI Assistant Agent Guide

This repository is a FastAPI + LangGraph AI chat app with browser-scoped RAG, web search, calculator tooling, and a lightweight frontend. The goal is to keep the system understandable, reliable, and easy to extend without turning the project into a messy monolith.

## Project purpose

The app helps users:

- chat in multiple conversation threads
- upload documents and ask grounded questions about them
- search the web for external or current information
- evaluate math problems through a calculator tool
- receive streaming answers in real time

## Core architecture

- `main.py` handles HTTP routes and SSE streaming.
- `chat_app_backend.py` contains the LangGraph workflow and the answer-validation loop.
- `chat_app_backend_rag.py` handles document parsing, embedding, retrieval, reranking, and context assembly.
- `tools.py` defines the assistant tools: document search, web search, calculator.
- `prompts.py` contains the system prompt and document-first behavior rules.
- `llm_router.py` manages fallback model routing through Groq.
- `static/` contains the browser UI and client-side rendering logic.

## Design principles

1. Prefer document-grounded answers when files are uploaded.
2. Treat uploaded files as the first source of truth, not as optional context.
3. Only use general knowledge when the docs are missing or insufficient.
4. Keep browser sessions isolated with a unique `kb_id` per user.
5. Prefer explicit tool routing and clear graph nodes over one monolithic prompt.
6. Keep retrieval quality higher than raw similarity by using reranking and relevance filtering.
7. Preserve a clean, maintainable frontend/backend split.

## Important repo conventions

- Do not break the per-browser document isolation model.
- Do not remove the answer-checking flow unless there is a strong reason and the replacement is tested.
- Keep document search behavior document-first and reranked from strongest to weakest chunk.
- Preserve SSE streaming behavior for real-time responses.
- Keep frontend and backend concerns separate.
- Avoid adding unnecessary dependencies.

## Functional workflow

### Chat flow

1. User sends a message from the browser.
2. FastAPI receives the request and starts a LangGraph stream.
3. `chat_message` invokes the LLM with available tools.
4. If a tool is needed, the graph calls the tool node and loops back.
5. If no tool is necessary, the response flows into validation.
6. The frontend renders streamed token output in real time.

### RAG flow

1. User uploads supported files.
2. Files are chunked and stored in a browser-specific Chroma collection.
3. The user asks a question.
4. Query is expanded and retrieved.
5. Relevant chunks are reranked and prioritized.
6. Only high-value context is passed to the final answer generation.

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Environment

Create a `.env` file with:

```env
GROQ_API_KEY=your_key_here
```

## Regression checklist before shipping changes

- chat still streams correctly
- uploaded document queries still use the correct collection
- answer validation still catches bad or irrelevant responses
- reranked document chunks keep the strongest result first
- browser session isolation still works

## Notes for future work

- add auth if this moves beyond prototype usage
- improve observability and tracing
- add automated tests around retrieval and graph routing
- consider longer-lived memory and better indexing strategies
