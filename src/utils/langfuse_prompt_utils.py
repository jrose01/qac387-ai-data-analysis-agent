from __future__ import annotations
import os
from typing import Any
from langfuse import get_client

langfuse = get_client()
DEFAULT_PROMPT_LABEL = os.getenv("LANGFUSE_PROMPT_LABEL", "production")


def get_prompt(name: str, label: str | None = None):
    return langfuse.get_prompt(name, label=label or DEFAULT_PROMPT_LABEL)


def compile_prompt(name: str, variables: dict[str, Any], label: str | None = None):
    prompt = get_prompt(name, label)
    return prompt, prompt.compile(**variables)
