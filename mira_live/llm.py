"""
LLM client for Mira Live — talks to LM Studio's OpenAI-compatible server.

Streams Hermes-3's reply token-by-token (so the UI types it out live and TTS can start early),
prepends the persona as the system message, and reports token usage so the UI can show a context
meter. Stateless: the caller (server) owns the conversation history per session.
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Optional

from openai import OpenAI

from . import config


class LLM:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
        self._persona = config.load_persona()
        self._model = config.LLM_MODEL or self._autodetect_model()

    def _autodetect_model(self) -> str:
        """LM Studio serves whatever model is loaded; grab its id so the caller doesn't have to
        match a name. Falls back to a sensible default if the server isn't up yet."""
        try:
            models = self._client.models.list()
            if models.data:
                mid = models.data[0].id
                print(f"[mira_live] using model: {mid}")
                return mid
        except Exception as e:
            print(f"[mira_live] couldn't list models ({e}); is LM Studio's server running on "
                  f"{config.LLM_BASE_URL}? Falling back to 'local-model'.")
        return "local-model"

    @property
    def model(self) -> str:
        return self._model

    @property
    def persona(self) -> str:
        return self._persona

    def reload_persona(self) -> None:
        self._persona = config.load_persona()

    def _messages(self, history: List[Dict]) -> List[Dict]:
        """persona (system) + the running conversation. History items are {role, content}."""
        msgs = [{"role": "system", "content": self._persona}]
        for m in history:
            role = m.get("role")
            if role in ("user", "assistant") and m.get("content"):
                msgs.append({"role": role, "content": m["content"]})
        return msgs

    def stream(self, history: List[Dict]):
        """Yield ('token', text) chunks as they arrive, then one final ('usage', {...}) with
        prompt/completion/total tokens (for the context meter). Yields ('error', msg) on failure."""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages(history),
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                presence_penalty=config.PRESENCE_PENALTY,
                frequency_penalty=config.FREQUENCY_PENALTY,
                max_tokens=config.MAX_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as e:
            yield ("error", f"{e}")
            return

        usage = None
        try:
            for chunk in resp:
                if getattr(chunk, "usage", None):
                    u = chunk.usage
                    usage = {
                        "prompt_tokens": getattr(u, "prompt_tokens", 0),
                        "completion_tokens": getattr(u, "completion_tokens", 0),
                        "total_tokens": getattr(u, "total_tokens", 0),
                    }
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None)
                    if piece:
                        yield ("token", piece)
        except Exception as e:
            yield ("error", f"{e}")
            return

        yield ("usage", usage or {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0})
