// Mira Live — chat UI client. Talks to the FastAPI backend: REST for sessions, WebSocket for
// streaming replies. Phase 1 = text chat; STT/TTS/avatar hook in here next.

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const inputEl = $("input");
const sendBtn = $("send");
const statusEl = $("status");
const sessionListEl = $("session-list");
const chatTitleEl = $("chat-title");

let state = {
  sessionId: null,
  ws: null,
  streaming: false,
  contextLimit: 8192,
  curMiraEl: null,
};

// ---------- WebSocket ----------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/chat`);
  ws.onopen = () => setStatus("");
  ws.onclose = () => { setStatus("disconnected — reconnecting…"); setTimeout(connect, 1500); };
  ws.onmessage = (ev) => handleServer(JSON.parse(ev.data));
  state.ws = ws;
}

function handleServer(msg) {
  switch (msg.type) {
    case "session": state.sessionId = msg.id; break;
    case "start":
      state.streaming = true; sendBtn.disabled = true; setStatus("Mira is typing…");
      state.curMiraEl = addMessage("mira", "");
      break;
    case "token":
      if (state.curMiraEl) { state.curMiraEl.body.textContent += msg.text; scrollDown(); }
      break;
    case "usage": setContext(msg.total_tokens, msg.limit); break;
    case "error":
      setStatus("⚠ " + msg.message);
      if (state.curMiraEl && !state.curMiraEl.body.textContent)
        state.curMiraEl.body.textContent = "(no response — is LM Studio's server running with a model loaded?)";
      break;
    case "end":
      // Snap the bubble to the cleaned final text (no *actions*/emoji that streamed raw).
      if (state.curMiraEl && msg.reply) state.curMiraEl.body.textContent = msg.reply;
      state.streaming = false; sendBtn.disabled = false; setStatus(""); state.curMiraEl = null;
      refreshSessions();
      break;
  }
}

// ---------- messages ----------
function addMessage(who, text) {
  const hint = messagesEl.querySelector(".empty-hint");
  if (hint) hint.remove();
  const el = document.createElement("div");
  el.className = `msg ${who}`;
  const label = document.createElement("div");
  label.className = "who";
  label.textContent = who === "user" ? "You" : "Mira";
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  el.appendChild(label); el.appendChild(body);
  messagesEl.appendChild(el);
  scrollDown();
  return { el, body };
}

function scrollDown() { messagesEl.scrollTop = messagesEl.scrollHeight; }
function setStatus(t) { statusEl.textContent = t; }

function setContext(total, limit) {
  if (limit) state.contextLimit = limit;
  const lim = state.contextLimit || 8192;
  const pct = Math.min(100, Math.round((total / lim) * 100));
  $("context-fill").style.width = pct + "%";
  $("context-label").textContent = `${total} / ${lim} tokens (${pct}%)` +
    (pct >= 85 ? " — getting full, consider New chat" : "");
}

// ---------- send ----------
async function send() {
  const text = inputEl.value.trim();
  if (!text || state.streaming) return;
  addMessage("user", text);
  inputEl.value = ""; autoGrow();
  // session_id may be null for a brand-new chat; the server creates it on first message
  // and tells us its id via a {type:"session"} reply.
  state.ws.send(JSON.stringify({ session_id: state.sessionId, text }));
}

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
sendBtn.addEventListener("click", send);
inputEl.addEventListener("input", autoGrow);
function autoGrow() { inputEl.style.height = "auto"; inputEl.style.height = Math.min(140, inputEl.scrollHeight) + "px"; }

// ---------- sessions ----------
$("new-chat").addEventListener("click", newChat);

function newChat() {
  // No POST — the session is created on the first message (avoids empty sessions).
  state.sessionId = null;
  chatTitleEl.textContent = "New chat";
  showEmptyState();
  setContext(0, state.contextLimit);
  highlightActive();
  inputEl.focus();
}

function showEmptyState() {
  messagesEl.innerHTML = `<div class="empty-hint">Start a new chat —<br>say something to Mira below.</div>`;
}

async function refreshSessions() {
  const list = await (await fetch("/api/sessions")).json();
  sessionListEl.innerHTML = "";
  for (const s of list) {
    const item = document.createElement("div");
    item.className = "session-item" + (s.id === state.sessionId ? " active" : "");
    item.dataset.id = s.id;
    const when = new Date((s.created || 0) * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    item.innerHTML = `
      <div class="session-main">
        <div class="session-title">${escapeHtml(s.title)}</div>
        <div class="meta">${when} · ${s.count} msgs</div>
      </div>
      <button class="session-del" title="Delete chat" aria-label="Delete chat">🗑</button>`;
    item.querySelector(".session-main").addEventListener("click", () => openSession(s.id));
    item.querySelector(".session-del").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteSession(s.id, s.title);
    });
    sessionListEl.appendChild(item);
  }
}

async function deleteSession(id, title) {
  if (!confirm(`Delete this chat?\n\n"${title}"\n\nThis can’t be undone.`)) return;
  try {
    await fetch(`/api/sessions/${id}`, { method: "DELETE" });
  } catch (e) { /* ignore; refresh below reflects reality */ }
  if (state.sessionId === id) {
    await newChat();          // the open chat was deleted -> start a fresh one
  } else {
    await refreshSessions();
  }
}

async function openSession(id) {
  const s = await (await fetch(`/api/sessions/${id}`)).json();
  if (s.error) return;
  state.sessionId = id;
  chatTitleEl.textContent = s.title || "Chat";
  messagesEl.innerHTML = "";
  for (const m of s.messages) {
    if (m.role === "user" || m.role === "assistant") addMessage(m.role === "user" ? "user" : "mira", m.content);
  }
  if (!s.messages || !s.messages.length) showEmptyState();
  setContext(s.tokens || 0, state.contextLimit);   // per-chat context, restored from disk
  highlightActive();
}

function highlightActive() {
  document.querySelectorAll(".session-item").forEach((el) =>
    el.classList.toggle("active", el.dataset.id === state.sessionId));
}

function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

// ---------- init ----------
async function init() {
  try {
    const meta = await (await fetch("/api/meta")).json();
    state.contextLimit = meta.context_limit || 8192;
    $("model-tag").textContent = "model: " + (meta.model || "—");
    setContext(0, state.contextLimit);
  } catch (e) { /* server warming */ }
  connect();
  // Don't create a session on load. Open the most recent chat, or show an empty state.
  const list = await (await fetch("/api/sessions")).json();
  if (list.length) {
    await openSession(list[0].id);   // list is newest-first
  } else {
    showEmptyState();
  }
  await refreshSessions();
}
init();
