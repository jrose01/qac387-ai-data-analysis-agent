from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class ToolResult:
    name: str
    text: str
    artifact_paths: List[str] = field(default_factory=list)
    structured: Optional[Dict[str, Any]] = None


def make_tool_result(
    name: str,
    text: str = "",
    artifact_paths: Optional[List[str]] = None,
    structured: Optional[Dict[str, Any]] = None,
    fallback_to_json: bool = True,
) -> ToolResult:
    artifact_paths = artifact_paths or []
    artifact_paths = [str(p) for p in artifact_paths]

    if not text and fallback_to_json and structured:
        try:
            text = json.dumps(structured, indent=2, default=str)
        except Exception:
            text = str(structured)

    if not text:
        text = "[No textual output returned by tool]"

    return ToolResult(
        name=name,
        text=text,
        artifact_paths=artifact_paths,
        structured=structured,
    )


def normalize_tool_return(tool_name: str, result: Any) -> ToolResult:
    """
    Normalize arbitrary tool outputs into a ToolResult.

    Accepted:
    - ToolResult
    - str
    - dict with optional keys: text, artifact_paths
    - tuple(text, artifact_paths)
    - any other object -> stringified
    """
    if isinstance(result, ToolResult):
        return result

    if isinstance(result, str):
        return make_tool_result(name=tool_name, text=result)

    if isinstance(result, dict):
        artifact_paths = result.get("artifact_paths", []) or []
        if not isinstance(artifact_paths, list):
            artifact_paths = [artifact_paths]
        artifact_paths = [str(p) for p in artifact_paths]

        if "text" in result and str(result.get("text", "")).strip():
            return make_tool_result(
                name=tool_name,
                text=str(result.get("text", "")),
                artifact_paths=artifact_paths,
                structured=result,
            )

        # Fallback: convert full dict to text
        return make_tool_result(
            name=tool_name,
            text="",
            artifact_paths=artifact_paths,
            structured=result,
            fallback_to_json=True,
        )

    if isinstance(result, tuple) and len(result) == 2:
        text, artifacts = result
        if artifacts is None:
            artifacts = []
        if not isinstance(artifacts, list):
            artifacts = [artifacts]
        return make_tool_result(
            name=tool_name,
            text=str(text),
            artifact_paths=[str(p) for p in artifacts],
        )

    return make_tool_result(name=tool_name, text=str(result))
