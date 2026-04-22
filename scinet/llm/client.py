from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.common import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL_NAME,
    extract_json_object,
    first_non_empty,
    get_env_value,
    load_env_values,
    write_text,
)


def load_llm_client(env_path: Path, params: dict[str, Any]) -> tuple[Any, str]:
    from openai import OpenAI

    env_values = load_env_values(env_path)
    api_key = first_non_empty(
        params.get("llm_api_key"),
        get_env_value(env_values, "OPENAI_API_KEY"),
    )
    if not api_key:
        raise ValueError(f"Missing LLM API key in {env_path}")
    base_url = first_non_empty(
        params.get("llm_base_url"),
        get_env_value(env_values, "OPENAI_BASE_URL"),
        DEFAULT_LLM_BASE_URL,
    )
    model_name = first_non_empty(
        params.get("llm_model_name"),
        params.get("llm_model"),
        get_env_value(env_values, "OPENAI_MODEL"),
        DEFAULT_LLM_MODEL_NAME,
    )
    return OpenAI(api_key=api_key, base_url=base_url), model_name


def call_llm_json(
    *,
    env_path: Path,
    params: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    artifact_path: Path | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    client, model_name = load_llm_client(env_path, params)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or "{}"
    if artifact_path is not None:
        write_text(artifact_path, content)
    return extract_json_object(content)
