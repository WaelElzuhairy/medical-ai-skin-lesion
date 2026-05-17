"""
Single LLM gateway for every agent in the system.

Provider-agnostic: defaults to Groq (free tier, Llama-3.3-70b) and falls
back to Anthropic (Claude Haiku) if LLM_PROVIDER=anthropic is set in .env.

Rules (non-negotiable per project plan):
  - NO agent may import `groq` or `anthropic` directly — use call_llm().
  - JSON mode is enforced when schema is provided.
  - Every call is logged with provider + model for traceability.
  - On parse failure, call_llm raises ValueError so the agent can retry once.
"""

from __future__ import annotations

import json
import time
from typing import Any

import config


def call_llm(
    system: str,
    user: str,
    schema: dict | None = None,
    temperature: float = 0.1,
) -> dict | str:
    """Call the configured LLM and return the response.

    Parameters
    ----------
    system:      System prompt for the agent.
    user:        User message / agent input.
    schema:      If provided, JSON mode is requested and the response is
                 parsed + returned as a dict. Raises ValueError on failure.
    temperature: Sampling temperature (default 0.1 for deterministic agents).

    Returns
    -------
    dict  — if schema was provided and parsing succeeded.
    str   — raw text if no schema was provided.
    """
    if config.LLM_PROVIDER == "groq":
        return _call_groq(system, user, schema, temperature)
    elif config.LLM_PROVIDER == "anthropic":
        return _call_anthropic(system, user, schema, temperature)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER!r}. Set to 'groq' or 'anthropic'.")


# ---------------------------------------------------------------------------
# Groq backend
# ---------------------------------------------------------------------------

def _call_groq(
    system: str,
    user: str,
    schema: dict | None,
    temperature: float,
) -> dict | str:
    from groq import Groq

    if not config.GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set in .env")

    client = Groq(api_key=config.GROQ_API_KEY)

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    kwargs: dict[str, Any] = {
        "model":       config.GROQ_MODEL,
        "messages":    messages,
        "max_tokens":  config.LLM_MAX_TOKENS,
        "temperature": temperature,
    }
    if schema is not None:
        kwargs["response_format"] = {"type": "json_object"}

    t0       = time.time()
    response = client.chat.completions.create(**kwargs)
    elapsed  = time.time() - t0
    content  = response.choices[0].message.content

    print(f"[LLM] groq/{config.GROQ_MODEL}  {elapsed:.1f}s  {len(content)} chars", flush=True)

    if schema is not None:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {content[:300]}") from e

    return content


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _call_anthropic(
    system: str,
    user: str,
    schema: dict | None,
    temperature: float,
) -> dict | str:
    import anthropic

    if not config.ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set in .env")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    sys_prompt = system
    if schema is not None:
        sys_prompt += "\n\nRespond with valid JSON only. No markdown, no explanation."

    t0 = time.time()
    message = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=sys_prompt,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
    )
    elapsed = time.time() - t0
    content = message.content[0].text

    print(f"[LLM] anthropic/{config.ANTHROPIC_MODEL}  {elapsed:.1f}s  {len(content)} chars", flush=True)

    if schema is not None:
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {content[:300]}") from e

    return content
