"""
Mira Live web server (FastAPI).

Serves the chat UI + avatar stage and brokers chat between the browser and LM Studio:
  GET  /                      -> the chatroom UI
  GET  /api/sessions          -> list past sessions (sidebar)
  POST /api/sessions          -> start a new chat (fresh context, persona re-applied)
  GET  /api/sessions/{id}     -> a session's messages (open old history)
  GET  /api/meta              -> model id + context limit (for the meter)
  WS   /ws/chat               -> {session_id, text} in; streamed reply tokens + usage out

Bind is 0.0.0.0 so the avatar stage is reachable on the LAN for OBS. Run via start_mira_live.bat.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, sessions
from .llm import LLM

_HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(_HERE, "web")

app = FastAPI(title="Mira Live")
llm = LLM()

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/meta")
def meta():
    return {"model": llm.model, "context_limit": config.CONTEXT_LIMIT}


@app.get("/api/sessions")
def api_list_sessions():
    return sessions.list_sessions()


@app.post("/api/sessions")
def api_new_session():
    return sessions.new_session()


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str):
    s = sessions.load(session_id)
    if s is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return s


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            session_id = data.get("session_id")
            text = (data.get("text") or "").strip()
            if not session_id or not text:
                continue

            s = sessions.load(session_id)
            if s is None:
                s = sessions.new_session()
                session_id = s["id"]
                await ws.send_json({"type": "session", "id": session_id})

            sessions.append(session_id, "user", text)
            history = sessions.load(session_id)["messages"]

            await ws.send_json({"type": "start"})
            reply_parts = []
            for kind, payload in llm.stream(history):
                if kind == "token":
                    reply_parts.append(payload)
                    await ws.send_json({"type": "token", "text": payload})
                elif kind == "usage":
                    total = (payload or {}).get("total_tokens", 0)
                    await ws.send_json({"type": "usage", "total_tokens": total,
                                        "limit": config.CONTEXT_LIMIT})
                elif kind == "error":
                    await ws.send_json({"type": "error", "message": payload})
            reply = "".join(reply_parts).strip()
            if reply:
                sessions.append(session_id, "assistant", reply)
            await ws.send_json({"type": "end", "reply": reply})
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def main():
    import uvicorn
    print("=" * 60)
    print(f"  Mira Live — chat UI + avatar stage")
    print(f"  Open:   http://localhost:{config.PORT}")
    print(f"  LAN/OBS: http://<this-laptop-ip>:{config.PORT}   (avatar stage at /avatar later)")
    print(f"  LLM:    {config.LLM_BASE_URL}  (start LM Studio's server with Hermes-3 loaded)")
    print("=" * 60)
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
