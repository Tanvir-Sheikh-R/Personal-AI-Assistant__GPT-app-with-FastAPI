// ---------------------------------------------------------------------------
// localStorage-backed state
//
// Threads and the kb_id live in localStorage, so chat history and uploaded
// documents persist per browser across refreshes, and different browsers never
// share each other's threads (each thread_id is a unique UUID).
// ---------------------------------------------------------------------------

function getKbId() {
  let kbId = localStorage.getItem("kb_id");
  if (!kbId) {
    kbId = crypto.randomUUID();
    localStorage.setItem("kb_id", kbId);
  }
  return kbId;
}
const kbId = getKbId();

function getThreads() {
  return JSON.parse(localStorage.getItem("threads") || "[]");
}

function saveThreads(threads) {
  localStorage.setItem("threads", JSON.stringify(threads));
}

function setActiveThreadId(id) {
  localStorage.setItem("active_thread_id", id);
}

function getActiveThreadId() {
  let id = localStorage.getItem("active_thread_id");
  const list = getThreads();
  if (!id || !list.find((t) => t.thread_id === id)) {
    id = createNewThread();
  }
  return id;
}

function createNewThread() {
  const id = crypto.randomUUID();
  const list = getThreads();
  list.unshift({ thread_id: id, title: null });
  saveThreads(list);
  setActiveThreadId(id);
  return id;
}

function updateThreadTitle(threadId, title) {
  const list = getThreads();
  const t = list.find((t) => t.thread_id === threadId);
  if (t) {
    t.title = title;
    saveThreads(list);
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

const messagesEl = document.getElementById("messages");
const messagesInner = document.getElementById("messages-inner");
const threadListEl = document.getElementById("thread-list");
const emptyStateEl = document.getElementById("empty-state");
const chatAreaEl = document.querySelector(".chat-area");
const composerWrapEl = document.querySelector(".composer-wrap");
const brandEl = document.getElementById("brand");

// The app's real logo (src/logo_green.svg), shown beside every assistant response.
const LOGO_SVG =
  '<svg class="avatar" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
  '<path fill="#6b8e55" d="m8,17v-5.6l-3.94,3.96c-.39.39-1.02.39-1.42,0h0c-.39-.4-.39-1.03,0-1.42l3.96-3.94H1c-.55,0-1-.45-1-1h0c0-.55.45-1,1-1h5.6l-3.96-3.94c-.39-.39-.39-1.02,0-1.42h0c.39-.4,1.03-.4,1.42,0l3.94,3.96V1c0-.55.45-1,1-1h0c.55,0,1,.45,1,1v5.6l3.94-3.96c.39-.39,1.02-.39,1.42,0h0c.39.4.39,1.03,0,1.42l-3.96,3.94h5.6c.55,0,1,.45,1,1h0c0,.55-.45,1-1,1h-5.6l3.96,3.94c.39.39.39,1.02,0,1.42h0c-.39.4-1.03.4-1.42,0l-3.94-3.96v5.6c0,.55-.45,1-1,1h0c-.55,0-1-.45-1-1Z"/>' +
  "</svg>";

const DOC_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zM13 9V3.5L18.5 9H13zM8 13h8v2H8v-2zm0-4h4v2H8V9z"/></svg>';

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------------------------------------------------------------------------
// Markdown configuration: KaTeX math + highlight.js code blocks (IDE style)
// ---------------------------------------------------------------------------

function renderMathToken(token) {
  if (window.katex) {
    try {
      const html = katex.renderToString(token.text, { displayMode: token.display, throwOnError: false });
      return token.display ? `<div class="math-block">${html}</div>` : html;
    } catch (_) {
      return escapeHtml(token.text);
    }
  }
  return escapeHtml(token.text);
}

if (window.marked) {
  // Block math: $$ ... $$
  const mathBlock = {
    name: "mathBlock",
    level: "block",
    start(src) {
      const m = src.match(/\$\$/);
      return m ? m.index : undefined;
    },
    tokenizer(src) {
      const m = src.match(/^\$\$([\s\S]+?)\$\$/);
      if (m) return { type: "mathBlock", raw: m[0], text: m[1], display: true };
      return undefined;
    },
    renderer(token) {
      return renderMathToken(token);
    },
  };

  // Inline math: $...$ without spaces (avoids mangling prices like "$249").
  const mathInline = {
    name: "mathInline",
    level: "inline",
    start(src) {
      const m = src.match(/\$/);
      return m ? m.index : undefined;
    },
    tokenizer(src) {
      const m = src.match(/^\$([^$\s\n]+)\$/);
      if (m) return { type: "mathInline", raw: m[0], text: m[1], display: false };
      return undefined;
    },
    renderer(token) {
      return renderMathToken(token);
    },
  };

  // Code fences become dark, IDE-style blocks with a language label + copy button.
  const codeRenderer = (codeArg, infostring) => {
    let code, lang;
    if (codeArg && typeof codeArg === "object") {
      code = codeArg.text || "";
      lang = (codeArg.lang || "").split(/\s+/)[0];
    } else {
      code = codeArg || "";
      lang = (infostring || "").split(/\s+/)[0];
    }

    let highlighted;
    try {
      const langId = lang && window.hljs && hljs.getLanguage(lang) ? lang : null;
      highlighted = langId
        ? hljs.highlight(code, { language: langId }).value
        : window.hljs
          ? hljs.highlightAuto(code).value
          : escapeHtml(code);
    } catch (_) {
      highlighted = escapeHtml(code);
    }

    const label = lang || "code";
    return (
      '<div class="code-block">' +
      `<div class="code-header"><span class="code-lang">${escapeHtml(label)}</span>` +
      '<button type="button" class="code-copy">Copy</button></div>' +
      `<pre><code class="hljs">${highlighted}</code></pre>` +
      "</div>"
    );
  };

  marked.use({ extensions: [mathBlock, mathInline], renderer: { code: codeRenderer } });
}

// Render assistant text as Markdown (falls back to plain text if marked is
// unavailable or the text is empty).
function renderBubble(el, text) {
  if (window.marked && text) {
    const cleanedText = text
      .replace(/[ \t]+\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    el.innerHTML = marked.parse(cleanedText, { breaks: false, gfm: true });
  } else {
    el.textContent = text;
  }
}

// Copy button for code blocks (event delegation — works for streamed content).
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".code-copy");
  if (!btn) return;
  const block = btn.closest(".code-block");
  const codeEl = block && block.querySelector("code");
  if (!codeEl) return;
  navigator.clipboard
    .writeText(codeEl.textContent)
    .then(() => {
      btn.textContent = "Copied!";
      setTimeout(() => {
        btn.textContent = "Copy";
      }, 1500);
    })
    .catch(() => {});
});

// True while we're loading an existing thread's history — prevents the hero
// "leave" animation from replaying on every thread switch.
let suppressHeroAnim = false;

function updateEmptyState() {
  const hasMessages = messagesInner.querySelectorAll(".message").length > 0;
  brandEl.hidden = false;

  if (hasMessages) {
    // Composer drops to the bottom once the conversation starts.
    if (composerWrapEl.parentElement !== chatAreaEl) chatAreaEl.appendChild(composerWrapEl);
    if (!suppressHeroAnim && !emptyStateEl.classList.contains("hidden")) {
      emptyStateEl.classList.add("leaving");
      setTimeout(() => {
        emptyStateEl.classList.add("hidden");
        emptyStateEl.classList.remove("leaving");
      }, 320);
    } else {
      emptyStateEl.classList.add("hidden");
      emptyStateEl.classList.remove("leaving");
    }
  } else {
    // Empty chat: the composer sits centered, right below the logo.
    if (composerWrapEl.parentElement !== emptyStateEl) emptyStateEl.appendChild(composerWrapEl);
    emptyStateEl.classList.remove("hidden");
    emptyStateEl.classList.remove("leaving");
  }
}

function renderThreadList() {
  const activeId = getActiveThreadId(); // must run first — creates the first thread if none exists
  threadListEl.innerHTML = "";

  getThreads().forEach((t) => {
    const row = document.createElement("div");
    row.className = "thread-row" + (t.thread_id === activeId ? " active" : "");

    const btn = document.createElement("button");
    btn.className = "thread-btn";
    btn.textContent = !t.title || !t.title.trim() ? "New chat" : t.title.trim();
    btn.onclick = () => {
      if (activeStreamController) activeStreamController.abort();
      setActiveThreadId(t.thread_id);
      renderThreadList();
      loadThread(t.thread_id);
    };

    const edit = document.createElement("button");
    edit.className = "thread-edit";
    edit.title = "Rename";
    edit.innerHTML =
      '<svg viewBox="0 0 24 24" width="14" height="14"><path fill="currentColor" d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>';
    edit.onclick = (e) => {
      e.stopPropagation();
      startRename(t, row);
    };

    row.appendChild(btn);
    row.appendChild(edit);
    threadListEl.appendChild(row);
  });
}

function startRename(thread, row) {
  const input = document.createElement("input");
  input.className = "thread-rename-input";
  input.value = thread.title || "";
  input.placeholder = "Chat name";
  row.innerHTML = "";
  row.appendChild(input);
  input.focus();
  input.select();

  let done = false;
  const commit = (save) => {
    if (done) return;
    done = true;

    const threads = getThreads();
    const item = threads.find((t) => t.thread_id === thread.thread_id);
    if (save && item) {
      const v = input.value.trim();
      item.title = v || null;
      saveThreads(threads);
    }

    renderThreadList();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit(true);
    else if (e.key === "Escape") commit(false);
  });
  input.addEventListener("blur", () => commit(true));
}

// Returns the (mutable) bubble element so streaming can update its text.
function appendMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  if (role === "assistant") {
    div.innerHTML = LOGO_SVG + '<div class="bubble"></div>';
  } else {
    div.innerHTML = '<div class="bubble"></div>';
  }
  const bubble = div.querySelector(".bubble");
  if (role === "assistant") {
    renderBubble(bubble, content);
  } else {
    bubble.textContent = content;
  }
  messagesInner.appendChild(div);
  updateEmptyState();
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

// User message that carries any attached documents as cards inside the bubble.
function appendUserMessage(text, filenames, error) {
  const div = document.createElement("div");
  div.className = "message user";
  let html = "";
  if (filenames && filenames.length) {
    html +=
      '<div class="attach-list">' +
      filenames.map((f) => `<div class="attach-card">${DOC_ICON}<span>${escapeHtml(f)}</span></div>`).join("") +
      "</div>";
  }
  if (text) html += `<div class="bubble">${escapeHtml(text)}</div>`;
  if (error) html += `<div class="attach-error">RAG indexing failed: ${escapeHtml(error)}</div>`;
  div.innerHTML = html;
  messagesInner.appendChild(div);
  updateEmptyState();
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

// Remove threads that no longer have any messages server-side (e.g. after a
// server restart wiped the in-memory conversations) so empty shells don't
// pile up in the sidebar. The active thread is always kept.
async function pruneStaleThreads(activeId) {
  const list = getThreads();
  const stale = [];
  for (const t of list) {
    if (t.thread_id === activeId) continue;
    try {
      const res = await fetch(`/threads/${t.thread_id}`);
      if (!res.ok) continue;
      const data = await res.json();
      if (!data.messages || !data.messages.length) stale.push(t.thread_id);
    } catch (_) {
      /* keep the thread if the server is unreachable */
    }
  }
  if (stale.length) {
    saveThreads(getThreads().filter((t) => !stale.includes(t.thread_id)));
    renderThreadList();
  }
}

// Clear only the message nodes, never the hero (so it can reappear).
function clearMessages() {
  messagesInner.innerHTML = "";
  updateEmptyState();
  messagesEl.scrollTop = 0;
}

let viewToken = 0;
async function loadThread(threadId) {
  const token = ++viewToken;
  messagesInner.innerHTML = "";
  emptyStateEl.classList.add("hidden");
  emptyStateEl.classList.remove("leaving");
  messagesEl.scrollTop = 0;

  const res = await fetch(`/threads/${threadId}`);
  if (token !== viewToken) return;  // a newer view (new chat / another switch) superseded this
  if (!res.ok) {
    updateEmptyState();
    return;
  }
  const data = await res.json();
  if (token !== viewToken) return;  // check again after the second await

  const threads = getThreads();
  const current = threads.find((t) => t.thread_id === threadId);
  if (current && (!data.messages || data.messages.length === 0)) {
    current.title = null;
    saveThreads(threads);
  }

  suppressHeroAnim = true;
  if (data.messages && data.messages.length) {
    data.messages.forEach((m) => appendMessage(m.role, m.content));
  }
  suppressHeroAnim = false;
  updateEmptyState();
}
// ---------------------------------------------------------------------------
// Sending a message + SSE streaming
// ---------------------------------------------------------------------------

const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const fileInput = document.getElementById("file-input");
const attachBtn = document.getElementById("attach-btn");
const attachChipsEl = document.getElementById("attach-chips");

let activeStreamController = null;
let pendingFiles = [];

function renderPendingFiles() {
  attachChipsEl.innerHTML = "";
  pendingFiles.forEach((f, i) => {
    const chip = document.createElement("span");
    chip.className = "attach-chip";
    chip.textContent = f.name;
    const x = document.createElement("button");
    x.type = "button";
    x.className = "chip-remove";
    x.textContent = "\u00d7";
    x.title = "Remove";
    x.onclick = () => {
      pendingFiles.splice(i, 1);
      renderPendingFiles();
    };
    chip.appendChild(x);
    attachChipsEl.appendChild(chip);
  });
}

attachBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  for (const f of fileInput.files) pendingFiles.push(f);
  fileInput.value = "";
  renderPendingFiles();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  const hasFiles = pendingFiles.length > 0;
  if (!text && !hasFiles) return;
  input.value = "";

  if (activeStreamController) activeStreamController.abort();
  activeStreamController = new AbortController();
  const { signal } = activeStreamController;

  const threadId = getActiveThreadId();
  const isFirstMessage = messagesInner.querySelectorAll(".message").length === 0;
  const filenames = pendingFiles.map((f) => f.name);

  // Show the user's message (text + attached docs) right away.
  const userDiv = appendUserMessage(text, filenames, null);

  // Index the attached documents so RAG can find them.
  let indexedError = null;
  if (hasFiles) {
    const fd = new FormData();
    fd.append("kb_id", kbId);
    for (const f of pendingFiles) fd.append("files", f);
    try {
      const upRes = await fetch("/upload", { method: "POST", body: fd });
      const up = await upRes.json();
      indexedError = up.success ? null : up.error;
    } catch (err) {
      indexedError = err.message;
    }
    if (indexedError) {
      const note = document.createElement("div");
      note.className = "attach-error";
      note.textContent = `RAG indexing failed: ${indexedError}`;
      userDiv.appendChild(note);
    }
  }
  pendingFiles = [];
  renderPendingFiles();

  const assistantBubble = appendMessage("assistant", "Thinking...");
  assistantBubble.classList.add("status-label");

  // Let the model know documents were just attached, so it uses the RAG tool
  // (the bubble itself shows the clean text + attachment cards).
  const chatText = filenames.length ? `Attached files: ${filenames.join(", ")}\n\n${text}` : text;
  const body = new URLSearchParams({ text: chatText, thread_id: threadId, kb_id: kbId });

  let res;
  try {
    res = await fetch("/chat", { method: "POST", body, signal });
  } catch (err) {
    if (err.name === "AbortError") return;
    throw err;
  }

  // Non-200 (e.g. 422 validation error): show the server's detail instead of
  // hanging on "Thinking...".
  if (!res.ok) {
    let detail = `Request failed (HTTP ${res.status})`;
    try {
      const errData = await res.json();
      if (errData && errData.detail) {
        detail = typeof errData.detail === "string" ? errData.detail : JSON.stringify(errData.detail);
      }
    } catch (_) {
      /* response wasn't JSON — keep the generic message */
    }
    assistantBubble.classList.remove("status-label");
    assistantBubble.textContent = `Something went wrong sending your message.\n${detail}`;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";

  while (true) {
    if (signal.aborted) return; // user switched threads mid-stream — stop writing here
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line.
    const events = buffer.split("\n\n");
    buffer = events.pop();

    for (const raw of events) {
      const eventLine = raw.split("\n").find((l) => l.startsWith("event:"));
      const dataLine = raw.split("\n").find((l) => l.startsWith("data:"));
      if (!eventLine || !dataLine) continue;

      const event = eventLine.replace("event:", "").trim();
      const data = dataLine.slice(6).replace(/\r$/, "").replace(/\\n/g, "\n");

      if (event === "clear") {
        // A check_answer correction replaces the previously streamed answer.
        answer = "";
        assistantBubble.classList.remove("status-label");
        assistantBubble.textContent = "";
      } else if (event === "token") {
        assistantBubble.classList.remove("status-label");
        answer += data;
        renderBubble(assistantBubble, answer);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      } else if (event === "status") {
        if (!answer) assistantBubble.textContent = `${data}...`;
      } else if (event === "done") {
        // stream finished
      }
    }
  }

  if (isFirstMessage) {
    const titleForm = new URLSearchParams({ first_message: text });
    const titleRes = await fetch(`/threads/${threadId}/title`, { method: "POST", body: titleForm });
    const titleData = await titleRes.json();
    updateThreadTitle(threadId, titleData.title);
    renderThreadList();
  }
});

// ---------------------------------------------------------------------------
// New chat / sidebar collapse
// ---------------------------------------------------------------------------

document.getElementById("new-chat-btn").addEventListener("click", () => {
  if (activeStreamController) activeStreamController.abort();
  viewToken++;

  const currentId = getActiveThreadId();
  const current = getThreads().find((t) => t.thread_id === currentId);

  if (!current || current.title === null) {
    // Already sitting on a blank, unused thread — don't create another one.
    clearMessages();
  } else {
    createNewThread();
    renderThreadList();
    clearMessages();
  }
  pendingFiles = [];
  renderPendingFiles();
  input.focus();
});

const appEl = document.querySelector(".app");
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarExpand = document.getElementById("sidebar-expand");

function setSidebar(collapsed) {
  appEl.classList.toggle("collapsed", collapsed);
  sidebarExpand.hidden = !collapsed;
  sidebarToggle.textContent = collapsed ? "\u00BB" : "\u00AB";
}

sidebarToggle.addEventListener("click", () => setSidebar(true));
sidebarExpand.addEventListener("click", (e) => {
  e.preventDefault();
  setSidebar(false);
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

renderThreadList();
const initialThread = getActiveThreadId();
loadThread(initialThread);
pruneStaleThreads(initialThread);

// ---------------------------------------------------------------------------
// Token Guard
// ---------------------------------------------------------------------------