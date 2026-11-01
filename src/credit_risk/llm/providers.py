"""Thin LLM provider adapters (all optional).

Each adapter exposes ``complete(system, user) -> str``. SDKs are imported lazily
so none of them is a hard dependency; only the configured provider needs to be
installed. If the call fails for any reason, the caller falls back to the
deterministic template, so the system never depends on an LLM being reachable.
"""
from __future__ import annotations

from credit_risk.config import Config


class LLMUnavailable(RuntimeError):
    pass


def complete(system: str, user: str, cfg: Config) -> str:
    provider = cfg.llm_provider
    model = str(cfg.get("llm.model"))
    max_tokens = int(cfg.get("llm.max_tokens", 700))
    temperature = float(cfg.get("llm.temperature", 0.2))

    if provider == "anthropic":
        return _anthropic(system, user, model, max_tokens, temperature, cfg)
    if provider == "openai":
        return _openai(system, user, model, max_tokens, temperature, cfg)
    if provider == "gemini":
        return _gemini(system, user, model, max_tokens, temperature, cfg)
    if provider == "ollama":
        return _ollama(system, user, model, max_tokens, temperature, cfg)
    raise LLMUnavailable(f"Unknown provider '{provider}'.")


def _anthropic(system, user, model, max_tokens, temperature, cfg) -> str:
    try:
        import anthropic
    except Exception as e:
        raise LLMUnavailable("anthropic SDK not installed") from e
    client = anthropic.Anthropic(api_key=cfg.llm_api_key)
    # Current Claude models reject sampling parameters (temperature/top_p), so
    # depth is controlled with effort instead; rephrasing needs very little.
    msg = client.messages.create(
        model=model, max_tokens=max_tokens,
        output_config={"effort": "low"},
        system=system, messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _openai(system, user, model, max_tokens, temperature, cfg) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise LLMUnavailable("openai SDK not installed") from e
    client = OpenAI(api_key=cfg.llm_api_key)
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""


def _gemini(system, user, model, max_tokens, temperature, cfg) -> str:
    try:
        import google.generativeai as genai
    except Exception as e:
        raise LLMUnavailable("google-generativeai SDK not installed") from e
    genai.configure(api_key=cfg.llm_api_key)
    gm = genai.GenerativeModel(model, system_instruction=system)
    resp = gm.generate_content(
        user,
        generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
    )
    return resp.text or ""


def _ollama(system, user, model, max_tokens, temperature, cfg) -> str:
    import json
    import urllib.request

    base = cfg.get("llm.ollama_base_url") or "http://localhost:11434"
    payload = {
        "model": model, "system": system, "prompt": user, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    req = urllib.request.Request(
        f"{base}/api/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()).get("response", "")
    except Exception as e:
        raise LLMUnavailable(f"Ollama call failed: {e}") from e
